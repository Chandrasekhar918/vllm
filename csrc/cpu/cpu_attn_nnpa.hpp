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
  bool try_init3ds(uint32_t s, uint32_t rows, uint32_t cols, const float* src) {
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &pre_desc, s, rows, cols);
    if (zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc) != ZDNN_OK) return false;
    if (zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt) != ZDNN_OK) return false;
    allocated = true;
    if (src && zdnn_transform_ztensor(&zt, src) != ZDNN_OK) return false;
    return true;
  }
  bool try_init2d(uint32_t rows, uint32_t cols, const float* src) {
    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &pre_desc, rows, cols);
    if (zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc) != ZDNN_OK) return false;
    if (zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt) != ZDNN_OK) return false;
    allocated = true;
    if (src && zdnn_transform_ztensor(&zt, src) != ZDNN_OK) return false;
    return true;
  }
  bool try_init1d(uint32_t cols, const float* src) {
    zdnn_init_pre_transformed_desc(ZDNN_1D, FP32, &pre_desc, cols);
    if (zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc) != ZDNN_OK) return false;
    if (zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt) != ZDNN_OK) return false;
    allocated = true;
    if (src && zdnn_transform_ztensor(&zt, src) != ZDNN_OK) return false;
    return true;
  }
  void store(float* dst) const {
    ZDNN_CHECK(zdnn_transform_origtensor(const_cast<zdnn_ztensor*>(&zt), dst), "store");
  }
  void free_buf() { if (allocated) { zdnn_free_ztensor_buffer(&zt); allocated = false; } }
  ~NNPATensor() { free_buf(); }
};

// ---------------------------------------------------------------------------
// TileGemmNNPA
//
// The framework calls gemm() once per block-group (QK) or head-dim-group (PV).
// The 8th parameter `block_size` carries the column count for each call:
//   QK: block_size = blocksize_alignment = 32  (KV token columns per group)
//   PV: block_size = block_size          = 32  (head-dim columns per group)
//
// `ldc` is the ROW STRIDE of C, not the column count.
// `ldb` is the ROW STRIDE of B.
//
// Both phases write exactly block_size (=32) columns per call.
// The caller advances the B and C pointers between calls to step through groups.
// ---------------------------------------------------------------------------
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
    const int32_t K    = (k_size > 0) ? k_size : dynamic_k_size;
    // Column count per call matches VXE's compile-time template param N:
    // QK: blocksize_alignment=32 KV cols; PV: headdim_alignment=32 output cols
    const int32_t N = (phase == AttentionGemmPhase::QK)
                    ? NNPA_BLOCK_SIZE_ALIGNMENT
                    : NNPA_HEAD_SIZE_ALIGNMENT;
    const int64_t lda_eff = (lda > 0) ? lda : (int64_t)K;
    const int64_t ldc_eff = (ldc > 0) ? ldc : (int64_t)N;

    if constexpr (phase == AttentionGemmPhase::QK) {
      // QK: A=[m, K=head_dim], B=[K, N=32 KV cols], C=[m, kv_tile_token_num]
      // Pack + clamp for DLF16, try NNPA matmul, fall back to CPU triple-loop.
      constexpr float DLF16_MAX = 500.0f;

      float* q_cont = (float*)malloc((size_t)m_size * K * sizeof(float));
      float* k_cont = (float*)malloc((size_t)K * N   * sizeof(float));
      float* bias   = (float*)calloc(N, sizeof(float));
      float* scores = (float*)malloc((size_t)m_size * N * sizeof(float));
      assert(q_cont && k_cont && bias && scores);

      for (int32_t i = 0; i < m_size; i++)
        for (int32_t d = 0; d < K; d++) {
          float v = a_tile[i * lda_eff + d];
          q_cont[i * K + d] = (v > DLF16_MAX) ? DLF16_MAX
                             : (v < -DLF16_MAX) ? -DLF16_MAX : v;
        }
      for (int32_t d = 0; d < K; d++)
        for (int32_t t = 0; t < N; t++) {
          float v = static_cast<float>(b_tile[d * ldb + t]);
          k_cont[d * N + t] = (v > DLF16_MAX) ? DLF16_MAX
                             : (v < -DLF16_MAX) ? -DLF16_MAX : v;
        }

      bool use_nnpa = true;
      NNPATensor tQ, tK, tBias, tOut;
      if (!tQ.try_init3ds(1, (uint32_t)m_size, (uint32_t)K, q_cont))    use_nnpa = false;
      if (use_nnpa && !tK.try_init2d((uint32_t)K, (uint32_t)N, k_cont)) use_nnpa = false;
      if (use_nnpa && !tBias.try_init1d((uint32_t)N, bias))             use_nnpa = false;
      if (use_nnpa && !tOut.try_init3ds(1, (uint32_t)m_size, (uint32_t)N, nullptr))
                                                                         use_nnpa = false;
      if (use_nnpa) {
        zdnn_status st = zdnn_matmul_bcast_op(
            &tQ.zt, &tK.zt, &tBias.zt, MATMUL_BCAST_OP_ADDITION, &tOut.zt);
        if (st == ZDNN_OK) tOut.store(scores);
        else               use_nnpa = false;
      }
      if (!use_nnpa) {
        for (int32_t i = 0; i < m_size; i++)
          for (int32_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (int32_t d = 0; d < K; d++)
              s += q_cont[i * K + d] * k_cont[d * N + j];
            scores[i * N + j] = s;
          }
      }

      for (int32_t i = 0; i < m_size; i++)
        for (int32_t j = 0; j < N; j++) {
          if (accum_c) c_tile[i * ldc_eff + j] += scores[i * N + j];
          else         c_tile[i * ldc_eff + j]  = scores[i * N + j];
        }

      ::free(scores); ::free(bias); ::free(k_cont); ::free(q_cont);

    } else {
      // PV: A=[m, K=curr_token_num], B=[K, N=32 head-dim cols], C=[m, head_dim]
      for (int32_t i = 0; i < m_size; i++)
        for (int32_t j = 0; j < N; j++) {
          float s = 0.0f;
          for (int32_t d = 0; d < K; d++)
            s += a_tile[i * lda_eff + d]
               * static_cast<float>(b_tile[d * ldb + j]);
          if (accum_c) c_tile[i * ldc_eff + j] += s;
          else         c_tile[i * ldc_eff + j]  = s;
        }
    }
  }
};

}  // namespace

template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t = scalar_t>
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
  static void copy_q_heads_tile(scalar_t* __restrict__ src, float* __restrict__ q_buffer,
                                const int32_t q_num, const int32_t q_heads_per_kv,
                                const int64_t q_num_stride, const int64_t q_head_stride,
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
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride,
      const float = 0.0f, const float = 0.0f) {
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx)
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) continue;
        const int64_t block_idx    = pos / block_size;
        const int64_t block_offset = pos % block_size;
        {
          const scalar_t* key_src = key + token_idx * key_token_num_stride
                                        + head_idx  * key_head_num_stride;
          scalar_t* key_dst = key_cache + block_idx  * num_blocks_stride
                                        + head_idx   * cache_head_num_stride
                                        + block_offset;
          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size)
            key_dst[j] = key_src[i];
        }
        {
          const scalar_t* val_src = value + token_idx * value_token_num_stride
                                          + head_idx  * value_head_num_stride;
          scalar_t* val_dst = value_cache + block_idx  * num_blocks_stride
                                          + head_idx   * cache_head_num_stride
                                          + block_offset * head_dim;
          std::memcpy(val_dst, val_src, sizeof(scalar_t) * head_dim);
        }
      }
  }
};

}  // namespace cpu_attention

#undef NNPA_BLOCK_SIZE_ALIGNMENT
#undef NNPA_HEAD_SIZE_ALIGNMENT
#undef NNPA_MAX_Q_HEAD_NUM_PER_ITER
#endif  // CPU_ATTN_NNPA_HPP
