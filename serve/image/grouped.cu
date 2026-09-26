// SPDX-License-Identifier: Apache-2.0
// Grouped (expert-batched) prefill GEMM for serialized ARVQ cold + NVFP4 hot
// experts on SM120/121. One launch per projection for all experts.
//
// MMA: mma.sync m16n8k16 f16 x f16 -> f32. A = routed activations (tokens are
// the M dimension), B = decoded weights (output channels are N).
//
// Weight decode never leaves registers: the serialized 16x64 tile layout is
// already the m16n8k64 FP4 fragment order (word j, lane = (row%8)*4 +
// (col%32)/8). Thread (q=lane/4, c=lane%4) therefore owns, for channel q (j
// even) / q+8 (j odd), eight consecutive physical k. We relabel K inside every
// 64-wide block with a fixed permutation so those eight k are exactly the B
// fragment slots of two m16n8k16 MMAs; the activations are stored in the same
// permuted order (x is permuted once per call; the intermediate h is written
// permuted by the gate/up epilogue). Contraction over K is invariant.
//
// Cold: w = alpha * s_fp16[ch, k/128] * (fp4(book[a]) + fp4(book[256 + r])).
//   The atom sum is exact in FP16; the block scale is applied to an FP32
//   partial every 128 k; alpha at the epilogue (same algebra as native).
// Hot:  w = hot_global[e, part] * e4m3[ch, k/16] * fp4 (exact in FP16).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

#define DEVI __device__ __forceinline__

DEVI void cvt_e2m1x8(uint32_t w, uint32_t (&h)[4]) {
  asm("{\n .reg .b8 b0,b1,b2,b3;\n mov.b32 {b0,b1,b2,b3}, %4;\n"
      " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(w));
}
DEVI uint32_t e4m3_h2(uint32_t byte) {
  uint32_t out;
  uint16_t v = (uint16_t)(byte | (byte << 8));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(v));
  return out;
}
DEVI uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEVI uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                   uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}
DEVI void cp_async16(uint32_t saddr, const void* g, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(saddr),
               "l"(g), "r"(bytes));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
DEVI void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
DEVI uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
// physical k within a 64 block -> logical (MMA) position.
DEVI int perm_logical(int p) {
  int within = p & 31, cc = within >> 3, t = ((p >> 5) << 1) + ((within & 7) >> 2),
      s = within & 3;
  return 16 * t + 2 * cc + (s < 2 ? s : 6 + s);
}

constexpr int XS = 64 + 8;  // smem row stride (halves): conflict-free ldmatrix

// MODE 0: gate/up (N = 2I rows), warp owns gate tile + matching up tile,
//         epilogue silu(g)*u -> permuted FP16 h [Rpad, I].
// MODE 1: down (N = hidden), warp owns 2 consecutive tiles, epilogue scales
//         by top-k weight and atomically adds into FP32 out [T, N].
// MODE 2: as MODE 1, but stores the weighted route row as BF16 into
//         out [T*topk, N] at row rid (deterministic; reduced by route_sum).
template <int MODE, int BM, int WARPS, int STAGES>
__global__ void __launch_bounds__(WARPS * 32)
    grouped_kernel(const __half* __restrict__ X, const int* __restrict__ routes,
                   const int* __restrict__ bexp, const int* __restrict__ lookups,
                   int n_experts, const uint16_t* __restrict__ cw,
                   const uint32_t* __restrict__ cb, const __half* __restrict__ cs,
                   const uint32_t* __restrict__ hw,
                   const uint32_t* __restrict__ hs, const float* __restrict__ hg,
                   float alpha, const float* __restrict__ tw, void* out, int K,
                   int N, int topk, int hot_parts, int sub) {
  constexpr int MT = BM / 16;
  __shared__ uint32_t lut[512];
  constexpr int XNEED = (BM * (WARPS * 32 + 4) * 4 + STAGES * XS * 2 - 1) / (STAGES * XS * 2);
  constexpr int XROWS = (MODE >= 1 && XNEED > BM) ? XNEED : BM;
  __shared__ __align__(16) __half xs[STAGES][XROWS][XS];
  __shared__ int srow[BM];
  // grid: x = route block * sub + channel CTA within slice, y = slice.
  const int m = blockIdx.x / sub;
  const int cx = blockIdx.y * sub + blockIdx.x % sub;
  const int ge = bexp[m];
  if (ge < 0) return;
  const int cold = lookups[ge], hot = lookups[n_experts + ge];
  if (cold < 0 && hot < 0) return;  // route owned by another path (defined as zero)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int NT16 = N / 16, KT = K / 64;
  int tiles[2];
  if (MODE == 0) {
    tiles[0] = cx * WARPS + warp;
    tiles[1] = tiles[0] + NT16 / 2;
  } else {
    tiles[0] = (cx * WARPS + warp) * 2;
    tiles[1] = tiles[0] + 1;
  }
  for (int i = tid; i < BM; i += WARPS * 32) {
    if (MODE == 0) {
      int rid = routes[m * BM + i];
      srow[i] = rid >= 0 ? rid / topk : -1;
    } else {
      srow[i] = m * BM + i;
    }
  }
  if (cold >= 0)
    for (int i = tid; i < 512; i += WARPS * 32) lut[i] = cb[(long long)cold * 512 + i];
  __syncthreads();

  auto load_stage = [&](int g) {
    if (g < KT) {
      int st = g % STAGES;
      for (int i = tid; i < BM * 8; i += WARPS * 32) {
        int r = i >> 3, ch = i & 7, row = srow[r];
        const __half* src = X + (long long)(row < 0 ? 0 : row) * K + g * 64 + ch * 8;
        cp_async16(smem_u32(&xs[st][r][ch * 8]), src, row < 0 ? 0 : 16);
      }
    }
    cp_commit();
  };

  float acc[MT][2][2][4];
  float tmp[MT][2][2][4];
#pragma unroll
  for (int a = 0; a < MT; a++)
#pragma unroll
    for (int b = 0; b < 2; b++)
#pragma unroll
      for (int n = 0; n < 2; n++)
#pragma unroll
        for (int v = 0; v < 4; v++) acc[a][b][n][v] = tmp[a][b][n][v] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) load_stage(s);

  // ldmatrix lane address components
  const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lk = (lane >> 4) * 8;
  const bool is_cold = cold >= 0;

  auto mainloop = [&](auto cold_tag) {
    constexpr bool COLD = decltype(cold_tag)::value;
    const int ex = COLD ? cold : hot;
    // raw weight registers for one k-step: cold = 16-bit index pairs,
    // hot = FP4 words (+ E4M3 scale words); prefetched one step ahead.
    uint32_t raw[2][4], rsw[2][2];
    auto fetch = [&](int g, uint32_t (&w)[2][4], uint32_t (&sw)[2][2]) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const long long tbase = ((long long)ex * NT16 + tiles[i]) * KT + g;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if constexpr (COLD)
            w[i][j] = cw[tbase * 128 + j * 32 + lane];
          else
            w[i][j] = hw[(tbase * 4 + j) * 32 + lane];
        }
        if constexpr (!COLD) {
#pragma unroll
          for (int h8 = 0; h8 < 2; h8++)
            sw[i][h8] = hs[((long long)ex * N + tiles[i] * 16 + q + 8 * h8) * KT + g];
        }
      }
    };
    fetch(0, raw, rsw);
    for (int g = 0; g < KT; g++) {
      uint32_t bfr[2][4][2][2];
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          uint32_t hv[4];
          const int nt = j & 1, tb = (j >> 1) * 2;
          if constexpr (COLD) {
            uint32_t hr[4];
            cvt_e2m1x8(lut[raw[i][j] & 255], hv);
            cvt_e2m1x8(lut[256 + ((raw[i][j] >> 8) & 255)], hr);
#pragma unroll
            for (int v = 0; v < 4; v++) hv[v] = hadd2(hv[v], hr[v]);
          } else {
            cvt_e2m1x8(raw[i][j], hv);
            const uint32_t sc = e4m3_h2((rsw[i][j & 1] >> (8 * ((j >> 1) * 2 + (c >> 1)))) & 255);
#pragma unroll
            for (int v = 0; v < 4; v++) hv[v] = hmul2(hv[v], sc);
          }
          bfr[i][tb][nt][0] = hv[0];
          bfr[i][tb][nt][1] = hv[1];
          bfr[i][tb + 1][nt][0] = hv[2];
          bfr[i][tb + 1][nt][1] = hv[3];
        }
      if (g + 1 < KT) fetch(g + 1, raw, rsw);
      cp_wait<STAGES - 2>();
      __syncthreads();
      load_stage(g + STAGES - 1);
      const int st = g % STAGES;
#pragma unroll
      for (int mi = 0; mi < MT; mi++) {
#pragma unroll
        for (int t = 0; t < 4; t++) {
          uint32_t a[4];
          ldsm_x4(a, smem_u32(&xs[st][mi * 16 + lrow][t * 16 + lk]));
#pragma unroll
          for (int i = 0; i < 2; i++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
              if constexpr (COLD)
                mma16816(tmp[mi][i][nt], a, bfr[i][t][nt][0], bfr[i][t][nt][1]);
              else
                mma16816(acc[mi][i][nt], a, bfr[i][t][nt][0], bfr[i][t][nt][1]);
            }
        }
      }
      if constexpr (COLD) {
        if (g & 1) {
          // end of a 128-k scale block: acc += tmp * fp16 scale[channel]
#pragma unroll
          for (int i = 0; i < 2; i++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
              const __half2 s2 = *reinterpret_cast<const __half2*>(
                  cs + (((long long)cold * NT16 + tiles[i]) * (K / 128) + (g >> 1)) * 16 + nt * 8 + 2 * c);
              float2 s = __half22float2(s2);
#pragma unroll
              for (int mi = 0; mi < MT; mi++) {
                acc[mi][i][nt][0] = fmaf(tmp[mi][i][nt][0], s.x, acc[mi][i][nt][0]);
                acc[mi][i][nt][1] = fmaf(tmp[mi][i][nt][1], s.y, acc[mi][i][nt][1]);
                acc[mi][i][nt][2] = fmaf(tmp[mi][i][nt][2], s.x, acc[mi][i][nt][2]);
                acc[mi][i][nt][3] = fmaf(tmp[mi][i][nt][3], s.y, acc[mi][i][nt][3]);
#pragma unroll
                for (int v = 0; v < 4; v++) tmp[mi][i][nt][v] = 0.f;
              }
            }
        }
      }
    }
  };
  if (is_cold)
    mainloop(std::true_type{});
  else
    mainloop(std::false_type{});
  cp_wait<0>();

  if constexpr (MODE == 0) {
    const int I = N / 2;
    const float sg = is_cold ? alpha : hg[(long long)hot * hot_parts];
    const float su = is_cold ? alpha : hg[(long long)hot * hot_parts + hot_parts - 1];
    __half* H = reinterpret_cast<__half*>(out);
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      const int ch = tiles[0] * 16 + nt * 8 + 2 * c;  // gate channel == h column
      const int col = (ch & ~63) + perm_logical(ch & 63);
#pragma unroll
      for (int mi = 0; mi < MT; mi++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          float hv[2];
#pragma unroll
          for (int v = 0; v < 2; v++) {
            float gv = __half2float(__float2half_rn(acc[mi][0][nt][hh * 2 + v] * sg));
            float uv = __half2float(__float2half_rn(acc[mi][1][nt][hh * 2 + v] * su));
            float sl = __half2float(__float2half_rn(gv / (1.f + __expf(-gv))));
            hv[v] = sl * uv;
          }
          const long long r = (long long)m * BM + mi * 16 + q + 8 * hh;
          *reinterpret_cast<__half2*>(H + r * I + col) = __floats2half2_rn(hv[0], hv[1]);
        }
    }
  } else {
    // Stage the CTA tile [BM, WARPS*32] (scaled by route weight) in shared
    // memory, then add rows into the FP32 output with coalesced 16-byte
    // atomics (one warp per row, 512 contiguous bytes).
    constexpr int CW = WARPS * 32, SS = CW + 4;
    float* stage = reinterpret_cast<float*>(&xs[0][0][0]);
    static_assert(sizeof(xs) >= (size_t)BM * SS * 4, "stage buffer");
    __syncthreads();
    const float sc = is_cold ? alpha : hg[(long long)hot * hot_parts];
#pragma unroll
    for (int mi = 0; mi < MT; mi++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = mi * 16 + q + 8 * hh;
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
          for (int nt = 0; nt < 2; nt++) {
            const int col = warp * 32 + i * 16 + nt * 8 + 2 * c;
            *reinterpret_cast<float2*>(&stage[r * SS + col]) =
                make_float2(acc[mi][i][nt][hh * 2], acc[mi][i][nt][hh * 2 + 1]);
          }
      }
    for (int i = tid; i < BM; i += WARPS * 32) {
      const int rid = routes[m * BM + i];
      srow[i] = rid;
    }
    __syncthreads();
    float* O = reinterpret_cast<float*>(out);
    const int ch0 = cx * CW;
    for (int r = warp; r < BM; r += WARPS) {
      const int rid = srow[r];
      if (rid < 0) continue;
      const float w = tw[rid] * sc;
      float* orow = O + (long long)(rid / topk) * N + ch0;
#pragma unroll
      for (int v = lane * 4; v < CW; v += 128) {
        float4 val = *reinterpret_cast<const float4*>(&stage[r * SS + v]);
        val.x *= w; val.y *= w; val.z *= w; val.w *= w;
        if constexpr (MODE == 1) {
          atomicAdd(reinterpret_cast<float4*>(orow + v), val);
        } else {
          __nv_bfloat162 lo = __floats2bfloat162_rn(val.x, val.y), hi = __floats2bfloat162_rn(val.z, val.w);
          uint2 pk = make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
          *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(out) + (long long)rid * N + ch0 + v) = pk;
        }
      }
    }
  }
}

template <int MODE, int BM, int WARPS, int STAGES>
static int launch(const void* X, const void* routes, const void* bexp,
                  const void* lookups, int n_experts, const void* cw,
                  const void* cb, const void* cs, const void* hw, const void* hs,
                  const void* hg, float alpha, const void* tw, void* out, int K,
                  int N, int topk, int hot_parts, int nblocks, void* stream, int sub) {
  if (K % 128 || N % 64) return (int)cudaErrorInvalidValue;
  int per = MODE == 0 ? WARPS * 16 : WARPS * 32;  // MODE 1/2 identical tiling
  int nx = MODE == 0 ? (N / 2) / per : N / per;
  if ((MODE == 0 ? (N / 2) : N) % per) return (int)cudaErrorInvalidValue;
  // sub: channel CTAs per slice (0 = all). Slices run in order, so the
  // concurrently-updated output columns stay L2-resident for MODE 1.
  if (sub <= 0 || sub > nx || nx % sub) sub = nx;
  grouped_kernel<MODE, BM, WARPS, STAGES><<<dim3(nblocks * sub, nx / sub), WARPS * 32, 0,
                                            (cudaStream_t)stream>>>(
      (const __half*)X, (const int*)routes, (const int*)bexp,
      (const int*)lookups, n_experts, (const uint16_t*)cw, (const uint32_t*)cb,
      (const __half*)cs, (const uint32_t*)hw, (const uint32_t*)hs,
      (const float*)hg, alpha, (const float*)tw, out, K, N, topk, hot_parts, sub);
  return (int)cudaGetLastError();
}

#define ARGS                                                                  \
  const void *X, const void *routes, const void *bexp, const void *lookups,   \
      int n_experts, const void *cw, const void *cb, const void *cs,          \
      const void *hw, const void *hs, const void *hg, float alpha,            \
      const void *tw, void *out, int K, int N, int topk, int hot_parts,       \
      int nblocks, void *stream, int sub
#define PASS X, routes, bexp, lookups, n_experts, cw, cb, cs, hw, hs, hg, alpha, \
             tw, out, K, N, topk, hot_parts, nblocks, stream, sub

// bm: tokens per route block (32/64/128). Returns cudaError_t.
extern "C" int arvq_grouped_gateup(int bm, ARGS) {
  switch (bm) {
    case 16: return launch<0, 16, 4, 3>(PASS);
    case 32: return launch<0, 32, 4, 3>(PASS);
    case 64: return launch<0, 64, 4, 3>(PASS);
    case 128: return launch<0, 128, 4, 3>(PASS);
  }
  return (int)cudaErrorInvalidValue;
}
extern "C" int arvq_grouped_down(int bm, ARGS) {
  switch (bm) {
    case 16: return launch<1, 16, 4, 3>(PASS);
    case 32: return launch<1, 32, 4, 3>(PASS);
    case 64: return launch<1, 64, 4, 3>(PASS);
    case 128: return launch<1, 128, 4, 3>(PASS);
  }
  return (int)cudaErrorInvalidValue;
}

extern "C" int arvq_grouped_down_store(int bm, ARGS) {
  switch (bm) {
    case 32: return launch<2, 32, 4, 3>(PASS);
    case 64: return launch<2, 64, 4, 3>(PASS);
  }
  return (int)cudaErrorInvalidValue;
}

// out[t, :] = sum_k y[t*topk + k, :] (BF16 in, FP32 sum, BF16 out). N % 8 == 0.
__global__ void route_sum_kernel(const __nv_bfloat16* __restrict__ y,
                                 __nv_bfloat16* __restrict__ out, int T, int N,
                                 int topk) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int per = N / 8;
  if (i >= (long long)T * per) return;
  const long long t = i / per;
  const int v = (int)(i % per) * 8;
  float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int k = 0; k < topk; k++) {
    uint4 pk = *reinterpret_cast<const uint4*>(y + (t * topk + k) * N + v);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&pk);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __bfloat1622float2(h[j]);
      s[2 * j] += f.x;
      s[2 * j + 1] += f.y;
    }
  }
  uint4 o;
  __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int j = 0; j < 4; j++) oh[j] = __floats2bfloat162_rn(s[2 * j], s[2 * j + 1]);
  *reinterpret_cast<uint4*>(out + t * N + v) = o;
}
extern "C" int arvq_route_sum(const void* y, void* out, int T, int N, int topk,
                              void* stream) {
  if (N % 8) return (int)cudaErrorInvalidValue;
  long long n = (long long)T * (N / 8);
  route_sum_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)y, (__nv_bfloat16*)out, T, N, topk);
  return (int)cudaGetLastError();
}

// xp[t, blk*64 + perm_logical(p)] = half(x[t, blk*64 + p]); one thread per
// (row, 64-block). x is BF16 (is_bf16=1) or FP16.
__global__ void permute_x_kernel(const uint16_t* __restrict__ x,
                                 __half* __restrict__ xp, long long rows, int K,
                                 int is_bf16) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int nb = K / 64;
  if (i >= rows * nb) return;
  const long long r = i / nb;
  const int b = (int)(i % nb);
  const uint16_t* src = x + r * K + b * 64;
  __align__(16) uint16_t in[64];
  __align__(16) __half o[64];
#pragma unroll
  for (int v = 0; v < 8; v++)
    *reinterpret_cast<uint4*>(&in[v * 8]) = *reinterpret_cast<const uint4*>(src + v * 8);
#pragma unroll
  for (int p = 0; p < 64; p++) {
    float f = is_bf16 ? __uint_as_float(((uint32_t)in[p]) << 16)
                      : __half2float(__ushort_as_half(in[p]));
    o[perm_logical(p)] = __float2half_rn(f);
  }
  __half* dst = xp + r * K + b * 64;
#pragma unroll
  for (int v = 0; v < 8; v++)
    *reinterpret_cast<uint4*>(dst + v * 8) = *reinterpret_cast<const uint4*>(&o[v * 8]);
}
extern "C" int arvq_permute_x(const void* x, void* xp, long long rows, int K,
                              int is_bf16, void* stream) {
  if (K % 64) return (int)cudaErrorInvalidValue;
  long long n = rows * (K / 64);
  permute_x_kernel<<<(unsigned)((n + 127) / 128), 128, 0, (cudaStream_t)stream>>>(
      (const uint16_t*)x, (__half*)xp, rows, K, is_bf16);
  return (int)cudaGetLastError();
}
