// SPDX-License-Identifier: Apache-2.0
// Copyright IBM Corp. 2025
//
// cpu_attn_nnpa.hpp
//
// NNPA attention backend for IBM Telum II.
// Replaces VXE SIMD intrinsics with zDNN API calls
// that offload to the Telum II on-chip AI accelerator.
//
// Operation mapping:
//   Q x K^T  ->  zdnn_matmul_bcast_op()  ->  NNPA_MATMUL_OP
//   P x V    ->  CPU fallback (V tensor layout mismatch with NNPA)
//
// Reference: cpu_attn_vxe.hpp (VXE/s390x SIMD)

#ifndef CPU_ATTN_NNPA_HPP
#define CPU_ATTN_NNPA_HPP

#include "cpu_attn_impl.hpp"
#include "/zDNN/zdnn/zdnn.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace cpu_attention {

namespace {

#define NNPA_BLOCK_SIZE_ALIGNMENT    32
#define NNPA_HEAD_SIZE_ALIGNMENT     32
#define NNPA_MAX_Q_HEAD_NUM_PER_ITER 16

// ─────────────────────────────────────────────────────────────────────────────
// ZDNN_CHECK macro
// Every zDNN API returns zdnn_status. ZDNN_OK (0) = success.
// Anything else = hardware or input error — print and abort.
// ─────────────────────────────────────────────────────────────────────────────
#define ZDNN_CHECK(call, msg)                                          \
  do {                                                                 \
    zdnn_status _st = (call);                                          \
    if (_st != ZDNN_OK) {                                              \
              (msg), (int)_st);                                        \
      abort();                                                         \
    }                                                                  \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// NNPATensor
//
// Wraps the full zdnn_ztensor lifecycle:
//   1. zdnn_init_pre_transformed_desc  — describe your float array shape
//   2. zdnn_generate_transformed_desc  — generate Telum II internal layout
//   3. zdnn_init_ztensor_with_malloc   — allocate 4KB-aligned buffer
//   4. zdnn_transform_ztensor          — convert FP32 → DLFLOAT16
//
// After computation:
//   5. zdnn_transform_origtensor       — convert result back → FP32
//   6. zdnn_free_ztensor_buffer        — release internal buffer
//
// Why DLFLOAT16?
//   Telum II NNPA operates in DLFLOAT16 (IBM 16-bit float).
//   zDNN handles the conversion transparently — you always pass FP32.
// ─────────────────────────────────────────────────────────────────────────────
struct NNPATensor {
  zdnn_ztensor     zt;
  zdnn_tensor_desc pre_desc;
  zdnn_tensor_desc tfrmd_desc;
  bool             allocated = false;

  // Initialize a 3DS tensor — shape [s, rows, cols]
  void init(uint32_t s, uint32_t rows, uint32_t cols,
            const float* src = nullptr, bool use_2d = false) {
    if (use_2d) {
      // ZDNN_2D layout for B matrix and bias
      zdnn_init_pre_transformed_desc(ZDNN_2D, FP32,
                                     &pre_desc, rows, cols);
    } else {
      // ZDNN_3DS layout for A, output
      zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32,
                                     &pre_desc, s, rows, cols);
    }
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc),
               "generate_transformed_desc");
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt),
               "init_ztensor_with_malloc");
    allocated = true;

    if (src != nullptr)
      ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "transform_ztensor");
  }

  // Initialize a 1D tensor — shape [cols] for bias
  void init_1d(uint32_t cols, const float* src = nullptr) {
    zdnn_init_pre_transformed_desc(ZDNN_1D, FP32, &pre_desc, cols);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc),
               "generate_transformed_desc_1d");
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt),
               "init_ztensor_with_malloc_1d");
    allocated = true;
    if (src != nullptr)
      ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "transform_ztensor_1d");
  }

  void load(const float* src) {
    ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "transform_ztensor (load)");
  }

  void store(float* dst) const {
    ZDNN_CHECK(zdnn_transform_origtensor(
                   const_cast<zdnn_ztensor*>(&zt), dst),
               "transform_origtensor (store)");
  }

  void free_buf() {
    if (allocated) {
      zdnn_free_ztensor_buffer(&zt);
      allocated = false;
    }
  }

  ~NNPATensor() { free_buf(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// nnpa_matmul
//
// Computes: C = A × B  (with zero bias)
// Offloads to Telum II via zdnn_matmul_op → NNPA_MATMUL_OP instruction.
//
// Parameters:
//   A    [s, m, k]  — first input  (Q tile or attention weights P)
//   B    [s, k, n]  — second input (K^T tile or V tile)
//   C    [s, m, n]  — output
//
// CRITICAL: zdnn_matmul_op requires a real bias tensor — NOT NULL.
// Passing NULL causes an illegal instruction crash on Telum II hardware.
// We always allocate a zero bias tensor of shape [s, 1, n].
// ─────────────────────────────────────────────────────────────────────────────
static void nnpa_matmul(const float* A, const float* B, float* C,
                        uint32_t s, uint32_t m, uint32_t k, uint32_t n,
                        int64_t lda=0, int64_t ldb=0, int64_t ldc=0) {
  NNPATensor tA, tB, tBias, tC;

  // Copy A with correct row strides into contiguous buffer
  float* a_buf = nullptr;
  if (lda > 0 && (int64_t)k != lda) {
    a_buf = (float*)malloc((size_t)m * k * sizeof(float));
    assert(a_buf != nullptr);
    for (uint32_t i = 0; i < m; i++)
      for (uint32_t j = 0; j < k; j++)
        a_buf[i * k + j] = A[i * lda + j];
    A = a_buf;
  }

  // Copy B into contiguous buffer
  // K cache layout: b_tile[dim * ldb + token] = K[dim, token]
  // We need b_buf[dim * n + token] = K[dim, token]
  float* b_buf = (float*)malloc((size_t)k * n * sizeof(float));
  assert(b_buf != nullptr);
  bool b_needs_free = true;
  if (ldb > 0 && (int64_t)n != ldb) {
    // Strided: copy row by row with stride ldb
    for (uint32_t i = 0; i < k; i++)
      for (uint32_t j = 0; j < n; j++)
        b_buf[i * n + j] = static_cast<float>(B[i * ldb + j]);
  } else {
    // Contiguous: simple copy
    for (size_t i = 0; i < (size_t)k * n; i++)
      b_buf[i] = static_cast<float>(B[i]);
  }
  B = b_buf;

  // B is already [k, n] row-major — zdnn reads it correctly

  }}

    for(uint32_t _h=0;_h<m;_h++){
    }
  tA.init(s, m, k, A);
  tB.init(s, k, n, B, true);
  tC.init(s, m, n);

  // Bias shape: [n] — zero values (ZDNN_1D)
  float* zero_bias = (float*)calloc(n, sizeof(float));
  assert(zero_bias != nullptr);
  tBias.init_1d(n, zero_bias);
  ::free(zero_bias);

  ZDNN_CHECK(
    zdnn_matmul_bcast_op(&tA.zt, &tB.zt, &tBias.zt,
                         MATMUL_BCAST_OP_ADDITION, &tC.zt),
    "zdnn_matmul_bcast_op"
  );

  // Extract result
  if (ldc > 0 && (int64_t)n != ldc) {
    float* c_tmp = (float*)malloc((size_t)m * n * sizeof(float));
    assert(c_tmp != nullptr);
    tC.store(c_tmp);
    for (uint32_t i = 0; i < m; i++)
      for (uint32_t j = 0; j < n; j++)
        C[i * ldc + j] = c_tmp[i * n + j];
    ::free(c_tmp);
  } else {
    tC.store(C);
  }

  if (a_buf) ::free(a_buf);
  if (b_buf) ::free(b_buf);

}

// ─────────────────────────────────────────────────────────────────────────────
// nnpa_softmax
//
// Computes row-wise softmax on a [rows, cols] float matrix.
// Offloads to Telum II via zdnn_softmax → NNPA_SOFTMAX instruction.
//
// zDNN softmax uses ZDNN_3DS layout [s, rows, cols] with s=1.
// Softmax normalizes across the innermost dimension (cols) —
// exactly what attention requires:
//   softmax(scores[q_head, :])  for each query head independently.
// ─────────────────────────────────────────────────────────────────────────────
static void nnpa_softmax(const float* input, float* output,
                         uint32_t rows, uint32_t cols) {
  NNPATensor tIn, tOut;
  tIn.init(1, rows, cols, input);
  tOut.init(1, rows, cols);

  // Fire softmax on Telum II
  // SOFTMAX_ACT_NONE = plain softmax (no extra activation)
  ZDNN_CHECK(
    zdnn_softmax(&tIn.zt, nullptr, SOFTMAX_ACT_NONE, &tOut.zt),
    "zdnn_softmax"
  );

  tOut.store(output);
}

// ─────────────────────────────────────────────────────────────────────────────
// TileGemmNNPA
//
// NNPA equivalent of TileGemmS390X from cpu_attn_vxe.hpp.
// Provides the gemm() template method required by AttentionMainLoop<>.
//
// Two phases:
//   QK phase: compute Q × K^T → attention scores
//   PV phase: compute P × V   → attention output
//
// Template parameter:
//   kv_cache_t — data type of K/V cache (float, BFloat16, Half, fp8)
// ─────────────────────────────────────────────────────────────────────────────
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
    // Resolve actual K at runtime
    const int32_t K = (k_size > 0) ? k_size : dynamic_k_size;
    {
      if (f) {
                (phase==AttentionGemmPhase::QK)?"QK":"PV",
                m_size, K, block_size, (long)lda, (long)ldb, (long)ldc, (int)accum_c, (void*)a_tile);
      }
    }

    // N = number of output columns
    // QK phase: N = block_size (number of KV tokens)
    // PV phase: N = HeadDimAlignment (32) — one head_dim group per call
    //           ldc=head_dim=64 is the OUTPUT stride, not the group size
    const int32_t N = (phase == AttentionGemmPhase::QK)
                          ? block_size
                          : NNPA_HEAD_SIZE_ALIGNMENT;

    // ── Convert B from kv_cache_t → float ─────────────────────────────────
    // zDNN always takes FP32. If KV cache is BFloat16 or Half, convert first.
    const size_t b_elems = (size_t)K * N;
    float* b_fp32 = nullptr;
    bool   b_needs_free = false;

    // Always copy B with correct strides — b_tile may not be contiguous
    b_fp32 = (float*)malloc(b_elems * sizeof(float));
    assert(b_fp32 != nullptr);
    b_needs_free = true;
    if constexpr (phase == AttentionGemmPhase::QK) {
      // K cache layout: b_tile[dim * ldb + token] = K[dim, token]
      // Copy to contiguous: b_fp32[dim * N + token] = K[dim, token]
      for (int32_t i = 0; i < K; i++)
        for (int32_t j = 0; j < N; j++)
          b_fp32[i * N + j] = static_cast<float>(b_tile[i * ldb + j]);
    } else {
      // V cache: b_tile[token*ldb + dim] = V[token, dim]
      for (int32_t i = 0; i < K; i++)
        for (int32_t j = 0; j < N; j++)
          b_fp32[i * N + j] = static_cast<float>(b_tile[i * ldb + j]);
    }

    // ── Scratch buffer for output (before accumulate) ──────────────────────
    // For PV: N=32 (group size) but ldc=64 (output stride) — use ldc for buffer
    const int64_t out_stride = (ldc > 0) ? ldc : N;
    const size_t c_elems = (size_t)m_size * out_stride;
    float* c_out = c_tile;
    float* c_tmp = nullptr;

    if (accum_c) {
      c_tmp = (float*)malloc(c_elems * sizeof(float));
      assert(c_tmp != nullptr);
      std::memset(c_tmp, 0, c_elems * sizeof(float));
      c_out = c_tmp;
    }

    if constexpr (phase == AttentionGemmPhase::QK) {
      // QK: use zdnn — K cache stored [dim,token] matches zdnn col-major
      nnpa_matmul(a_tile, b_fp32, c_out, 1,
                  (uint32_t)m_size, (uint32_t)K, (uint32_t)N, lda, N, ldc);
    } else {
      // PV: CPU fallback — V stored [token,dim] row-major
      // N=32 (head_dim group), ldc=64 (output stride in partial_q_buffer)
      uint32_t _m=(uint32_t)m_size, _k=(uint32_t)K, _n=(uint32_t)N;
      int64_t _lda=lda>0?lda:_k;
      int64_t _ldc=ldc>0?ldc:_n;
      for(uint32_t _i=0;_i<_m;_i++)
        for(uint32_t _j=0;_j<_n;_j++) {
          float _s=0;
          for(uint32_t _d=0;_d<_k;_d++)
            _s += a_tile[_i*_lda+_d] * b_fp32[_d*_n+_j];
          c_out[_i*_ldc+_j] = _s;
        }
    }

    // ── Accumulate if needed (C += C_new) ─────────────────────────────────
    if (accum_c) {
      for (size_t i = 0; i < c_elems; ++i)
        c_tile[i] += c_out[i];
      ::free(c_tmp);
    }

    if (b_needs_free)
      ::free(b_fp32);
  }
};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// AttentionImpl<ISA::NNPA>
//
// Full attention implementation for Telum II NNPA.
// Plugs into the vLLM AttentionMainLoop<> framework.
//
// Attention formula (Flash Attention style, tiled):
//   For each KV tile:
//     scores  = Q_tile × K^T_tile     → zdnn_matmul_op (NNPA)
//     scores *= scale                  → CPU scalar multiply
//     weights = softmax(scores)        → zdnn_softmax (NNPA)
//     output += weights × V_tile       → zdnn_matmul_op (NNPA)
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::NNPA, scalar_t, head_dim> {
 public:
  // ── Type aliases (required by AttentionMainLoop) ─────────────────────────
  using query_t                 = scalar_t;
  using q_buffer_t              = float;
  using kv_cache_t              = scalar_t;
  using logits_buffer_t         = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t           = float;

  // ── Constants (required by AttentionMainLoop) ────────────────────────────
  constexpr static int64_t BlockSizeAlignment      = NNPA_BLOCK_SIZE_ALIGNMENT;
  constexpr static int64_t HeadDimAlignment        = NNPA_HEAD_SIZE_ALIGNMENT;
  constexpr static int64_t MaxQHeadNumPerIteration = NNPA_MAX_Q_HEAD_NUM_PER_ITER;
  constexpr static int64_t HeadDim                 = head_dim;
  constexpr static ISA     ISAType                 = ISA::NNPA;
  // Scale is applied to Q before matmul (not on logits after)
  constexpr static bool    scale_on_logits         = false;

 public:
  AttentionImpl() {}

  // ── execute_attention ─────────────────────────────────────────────────────
  // Plug TileGemmNNPA into AttentionMainLoop<>.
  // VXE uses TileGemmS390X — we use TileGemmNNPA.
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    // Confirm NNPA is active
    {static bool _printed=false; if(!_printed){
      _printed=true;}}
    attention<TileGemmNNPA<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // ── KV cache stride helpers (same layout as VXE) ─────────────────────────
  constexpr static int64_t k_cache_token_group_stride(const int32_t) {
    return BlockSizeAlignment;
  }

  constexpr static int64_t v_cache_token_group_stride(const int32_t) {
    return head_dim * BlockSizeAlignment;
  }

  constexpr static int64_t v_cache_head_group_stride(const int32_t) {
    return HeadDimAlignment;
  }

  // ── copy_q_heads_tile ─────────────────────────────────────────────────────
  // Copy Q tile from model input → float buffer with scale applied.
  // VXE uses SIMD (vec_xl, vec_mul). We use a plain loop — correct and
  // portable. The Q copy is not on the critical path vs matmul.
  static void copy_q_heads_tile(scalar_t* __restrict__ src,
                                float*    __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride,
                                float scale) {
    for (int32_t i = 0; i < q_num; ++i) {
      for (int32_t h = 0; h < q_heads_per_kv; ++h) {
        const scalar_t* curr_src =
            src + i * q_num_stride + h * q_head_stride;
        float* curr_dst =
            q_buffer + i * q_heads_per_kv * head_dim + h * head_dim;

        for (int64_t d = 0; d < head_dim; ++d)
          curr_dst[d] = static_cast<float>(curr_src[d]) * scale;
      }
    }
              q_num,q_heads_per_kv,(long)q_num_stride,(long)q_head_stride,scale,(void*)q_buffer,(void*)src);
  }

  // ── reshape_and_cache ─────────────────────────────────────────────────────
  // Store K/V into paged KV cache.
  // Layout is identical to VXE — copied exactly.
  static void reshape_and_cache(
      const scalar_t* __restrict__ key,
      const scalar_t* __restrict__ value,
      scalar_t*       __restrict__ key_cache,
      scalar_t*       __restrict__ value_cache,
      const int64_t*  __restrict__ slot_mapping,
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
      const int64_t block_size_stride) {
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) continue;

        const int64_t block_idx    = pos / block_size;
        const int64_t block_offset = pos % block_size;

        // Key cache
        {
          const scalar_t* key_src =
              key + token_idx * key_token_num_stride +
              head_idx * key_head_num_stride;
          scalar_t* key_dst =
              key_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride + block_offset;

          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size)
            key_dst[j] = key_src[i];
          if (token_idx == 0 && head_idx == 0) {
            if (f) {
              for (int i = 0; i < 8; i++)
            }
          }
        }

        // Value cache
        {
          const scalar_t* val_src =
              value + token_idx * value_token_num_stride +
              head_idx * value_head_num_stride;
          scalar_t* val_dst =
              value_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride +
              block_offset * head_dim;

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
