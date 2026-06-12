#ifndef CPU_ATTN_NNPA_HPP
#define CPU_ATTN_NNPA_HPP

#include "cpu_attn_impl.hpp"
#include "/zDNN/zdnn/zdnn.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace cpu_attention {

namespace {

#define NNPA_BLOCK_SIZE_ALIGNMENT    32
#define NNPA_HEAD_SIZE_ALIGNMENT     32
#define NNPA_MAX_Q_HEAD_NUM_PER_ITER 16

#define ZDNN_CHECK(call, msg)                                        \
  do {                                                               \
    zdnn_status _st = (call);                                        \
    if (_st != ZDNN_OK) {                                            \
      fprintf(stderr, "[NNPA] zDNN error at %s: status=%d\n",       \
              (msg), (int)_st);                                      \
      abort();                                                       \
    }                                                                \
  } while (0)

struct NNPATensor {
  zdnn_ztensor     zt;
  zdnn_tensor_desc pre_desc;
  zdnn_tensor_desc tfrmd_desc;
  bool             allocated = false;

  void init3ds(uint32_t s, uint32_t rows, uint32_t cols, const float* src) {
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &pre_desc, s, rows, cols);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc), "gen_3ds");
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt), "malloc_3ds");
    allocated = true;
    if (src) ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "xform_3ds");
  }
  void init2d(uint32_t rows, uint32_t cols, const float* src) {
    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &pre_desc, rows, cols);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc), "gen_2d");
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt), "malloc_2d");
    allocated = true;
    if (src) ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "xform_2d");
  }
  void init1d(uint32_t cols, const float* src) {
    zdnn_init_pre_transformed_desc(ZDNN_1D, FP32, &pre_desc, cols);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc), "gen_1d");
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt), "malloc_1d");
    allocated = true;
    if (src) ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "xform_1d");
  }
  void store(float* dst) const {
    ZDNN_CHECK(zdnn_transform_origtensor(const_cast<zdnn_ztensor*>(&zt), dst), "store");
  }
  void free_buf() {
    if (allocated) { zdnn_free_ztensor_buffer(&zt); allocated = false; }
  }
  ~NNPATensor() { free_buf(); }
};

template <typename kv_cache_t>
class TileGemmNNPA {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile,
                                const int64_t lda,
                                const int64_t ldb,
                                const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    const int32_t K = (k_size > 0) ? k_size : dynamic_k_size;
    const int32_t N = (phase == AttentionGemmPhase::QK)
                          ? block_size : NNPA_HEAD_SIZE_ALIGNMENT;
    const int64_t lda_eff = (lda > 0) ? lda : (int64_t)K;
    const int64_t ldc_eff = (ldc > 0) ? ldc : (int64_t)N;
    // For QK: ldc_eff = kv_tile_token_num = actual valid token count
    const int32_t valid_n = (int32_t)ldc_eff;

    if constexpr (phase == AttentionGemmPhase::QK) {
      // Pack Q contiguous [m x K]
      float* q_cont = (float*)malloc((size_t)m_size * K * sizeof(float));
      assert(q_cont);
      for (int32_t i = 0; i < m_size; i++)
        for (int32_t d = 0; d < K; d++)
          q_cont[i * K + d] = a_tile[i * lda_eff + d];

      // Pack K: cache layout [dim * ldb + token] -> contiguous [K x valid_n]
      float* k_cont = (float*)malloc((size_t)K * valid_n * sizeof(float));
      assert(k_cont);
      for (int32_t d = 0; d < K; d++)
        for (int32_t t = 0; t < valid_n; t++)
          k_cont[d * valid_n + t] = static_cast<float>(b_tile[d * ldb + t]);

      float* bias = (float*)calloc(valid_n, sizeof(float));
      assert(bias);

      // scores[m x valid_n] contiguous output
      float* scores = (float*)malloc((size_t)m_size * valid_n * sizeof(float));
      assert(scores);

      NNPATensor tQ, tK, tBias, tOut;
      tQ.init3ds(1, (uint32_t)m_size, (uint32_t)K, q_cont);
      tK.init2d((uint32_t)K, (uint32_t)valid_n, k_cont);
      tBias.init1d((uint32_t)valid_n, bias);
      tOut.init3ds(1, (uint32_t)m_size, (uint32_t)valid_n, nullptr);

      ZDNN_CHECK(zdnn_matmul_bcast_op(&tQ.zt, &tK.zt, &tBias.zt,
                                       MATMUL_BCAST_OP_ADDITION, &tOut.zt),
                 "zdnn_matmul_bcast_op QK");
      tOut.store(scores);

      // Write scores into c_tile[m x ldc_eff]
      for (int32_t i = 0; i < m_size; i++)
        for (int32_t j = 0; j < valid_n; j++) {
          if (accum_c)
            c_tile[i * ldc_eff + j] += scores[i * valid_n + j];
          else
            c_tile[i * ldc_eff + j]  = scores[i * valid_n + j];
        }

      ::free(scores);
      ::free(bias);
      ::free(k_cont);
      ::free(q_cont);

    } else {
      // PV: CPU fallback (V is row-major, NNPA needs col-major)
      for (int32_t i = 0; i < m_size; i++)
        for (int32_t j = 0; j < valid_n; j++) {
          float s = 0.0f;
          for (int32_t d = 0; d < K; d++)
            s += a_tile[i * lda_eff + d] * static_cast<float>(b_tile[d * ldb + j]);
          if (accum_c)
            c_tile[i * ldc_eff + j] += s;
          else
            c_tile[i * ldc_eff + j]  = s;
        }
    }
  }
};

}  // namespace

template <typename scalar_t, int64_t head_dim,
          typename kv_cache_scalar_t = scalar_t>
class AttentionImpl<ISA::NNPA, scalar_t, head_dim, kv_cache_scalar_t> {
 public:
  using query_t                 = scalar_t;
  using q_buffer_t              = float;
  using kv_cache_t              = scalar_t;
  using logits_buffer_t         = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t           = float;

  constexpr static int64_t BlockSizeAlignment      = NNPA_BLOCK_SIZE_ALIGNMENT;
  constexpr static int64_t HeadDimAlignment        = NNPA_HEAD_SIZE_ALIGNMENT;
  constexpr static int64_t MaxQHeadNumPerIteration = NNPA_MAX_Q_HEAD_NUM_PER_ITER;
  constexpr static int64_t HeadDim                 = head_dim;
  constexpr static ISA     ISAType                 = ISA::NNPA;
  constexpr static bool    scale_on_logits         = false;

  AttentionImpl() {}

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    static bool _printed = false;
    if (!_printed) {
      fprintf(stderr, "[NNPA] Attention running on Telum II NNPA via zdnn\n");
      _printed = true;
    }
    attention<TileGemmNNPA<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  constexpr static int64_t k_cache_token_group_stride(const int32_t) { return BlockSizeAlignment; }
  constexpr static int64_t v_cache_token_group_stride(const int32_t) { return head_dim * BlockSizeAlignment; }
  constexpr static int64_t v_cache_head_group_stride(const int32_t)  { return HeadDimAlignment; }

  static void copy_q_heads_tile(scalar_t* __restrict__ src,
                                float* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride,
                                float scale) {
    for (int32_t i = 0; i < q_num; ++i)
      for (int32_t h = 0; h < q_heads_per_kv; ++h) {
        const scalar_t* s = src + i * q_num_stride + h * q_head_stride;
        float* d = q_buffer + i * q_heads_per_kv * head_dim + h * head_dim;
        for (int64_t dim = 0; dim < head_dim; ++dim)
          d[dim] = static_cast<float>(s[dim]) * scale;
      }
  }

  static void reshape_and_cache(
      const scalar_t* __restrict__ key,
      const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache,
      scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping,
      const int64_t token_num,
      const int64_t key_token_num_stride,
      const int64_t value_token_num_stride,
      const int64_t head_num,
      const int64_t key_head_num_stride,
      const int64_t value_head_num_stride,
      const int64_t num_blocks,
      const int64_t num_blocks_stride,
      const int64_t cache_head_num_stride,
      const int64_t block_size,
      const int64_t block_size_stride,
      const float = 0.0f,
      const float = 0.0f) {
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) continue;
        const int64_t block_idx    = pos / block_size;
        const int64_t block_offset = pos % block_size;
        {
          const scalar_t* key_src = key + token_idx * key_token_num_stride + head_idx * key_head_num_stride;
          scalar_t* key_dst = key_cache + block_idx * num_blocks_stride + head_idx * cache_head_num_stride + block_offset;
          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size)
            key_dst[j] = key_src[i];
        }
        {
          const scalar_t* val_src = value + token_idx * value_token_num_stride + head_idx * value_head_num_stride;
          scalar_t* val_dst = value_cache + block_idx * num_blocks_stride + head_idx * cache_head_num_stride + block_offset * head_dim;
          std::memcpy(val_dst, val_src, sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};

}  // namespace cpu_attention

#undef NNPA_BLOCK_SIZE_ALIGNMENT
#undef NNPA_HEAD_SIZE_ALIGNMENT
#undef NNPA_MAX_Q_HEAD_NUM_PER_ITER

#endif  // CPU_ATTN_NNPA_HPP
