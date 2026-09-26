# SPDX-License-Identifier: Apache-2.0
"""Expert-batched prefill for serialized ARVQ (cold) + NVFP4 (hot) experts.

Drop-in replacement for the grouped prefill entry used by ``arvq_mlp``:

    batched_cold_prefill(x, topk_weights, topk_ids, lookups, tensors, alphas,
                         projection=None, chunk_tokens=128) -> [T, hidden]

All routes (cold and hot) of one call run as three CUDA launches plus a few
sync-free torch ops, independent of the number of experts:

  0. routing prep (torch): stable sort of the T*top_k routes by global expert,
     each expert's list padded to BLOCK_M; x permuted/cast to FP16 once.
  1. arvq_grouped_gateup: per (route block, 64 gate + 64 up channels) CTA,
     decode w13 in registers (cold: per-expert 2x FP4 codebook atoms from a
     shared-memory LUT via cvt.f16x2.e2m1x2; hot: NVFP4 * E4M3), FP16 mma.sync
     with FP32 accumulate, block scales applied in FP32, silu(g)*u -> FP16.
  2. arvq_grouped_down: w2 likewise; epilogue scales by the top-k weight and
     adds rows into an FP32 [T, hidden] output with 16-byte vector atomics,
     grid ordered by 512-channel slices so the updated columns stay in L2.
     (``down="store"`` instead writes BF16 route rows + a reduction kernel:
     deterministic, same speed, ~3x larger rounding error.)

Numerics: decoded weights are exact in FP16; x is converted to FP16 exactly as
the native path does before packing activation planes. Unlike native, no
4-plane FP4 activation approximation is used, so the result is at least as
close to the FP32 reference of the serialized weights.

Graph/compile safety: no host synchronization, no data-dependent shapes
(grid sized from T, top_k and #experts); ctypes launches use the current
torch stream, so it is CUDA-graph capturable. It is only called from inside
the opaque ``arvq_hybrid::mlp`` custom op, so torch.compile never traces it.

Not supported (``supported()`` is False): mcbook16 (v5) projections.
"""

import ctypes
import os
from pathlib import Path

import torch

_LIB = None
_PERM = {}
_CHUNK_TOKENS = int(os.environ.get("VLLM_ARVQ_BATCHED_MAX_TOKENS", "16384"))


def _lib():
    global _LIB
    if _LIB is None:
        default = Path(__file__).with_name("arvq") / "grouped.so"
        if not default.is_file():
            default = Path(__file__).with_name("cuda") / "grouped.so"
        path = os.environ.get("VLLM_ARVQ_GROUPED_LIB", str(default))
        lib = ctypes.CDLL(path)
        args = ([ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_int]
                + [ctypes.c_void_p] * 6 + [ctypes.c_float] + [ctypes.c_void_p] * 2
                + [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_int])
        for name in ("arvq_grouped_gateup", "arvq_grouped_down", "arvq_grouped_down_store"):
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
        lib.arvq_route_sum.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.arvq_route_sum.restype = ctypes.c_int
        lib.arvq_permute_x.argtypes = ([ctypes.c_void_p] * 2 + [ctypes.c_longlong]
                                       + [ctypes.c_int] * 2 + [ctypes.c_void_p])
        lib.arvq_permute_x.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _check(err, what):
    if err:
        raise RuntimeError(f"{what} failed with CUDA error {err}")


def supported(tensors) -> bool:
    """v4 (and v3) layers; mcbook16 selectors/factors must be empty."""
    return (
        len(tensors) == 16
        and all(t.numel() == 0 for t in (tensors[6], tensors[7], tensors[14], tensors[15]))
        and tensors[1].dtype == torch.uint32 and tensors[1].ndim == 2
        and tensors[1].shape[1] == 512
        and tensors[2].dtype == torch.float16 and tensors[10].dtype == torch.float16
    )


def _route_blocks(topk_ids, n_experts, bm):
    """Expert-major, BLOCK_M-padded route order without host synchronization.

    Returns routes [nblocks*bm] int32 (flat route id = token*top_k + k, -1 pad),
    block expert [nblocks] int32 (global expert id, -1 = unused), nblocks.
    """
    R = topk_ids.numel()
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(n_experts, dtype=torch.long, device=dev)
    counts.index_add_(0, flat, torch.ones_like(flat))
    padded = (counts + bm - 1) // bm * bm
    pad_end = torch.cumsum(padded, 0)
    start = torch.cumsum(counts, 0) - counts
    sorted_e = flat[order]
    pos = (pad_end - padded)[sorted_e] + (torch.arange(R, device=dev) - start[sorted_e])
    nblocks = (R + bm - 1) // bm + n_experts
    routes = torch.full((nblocks * bm,), -1, dtype=torch.int32, device=dev)
    routes[pos] = order.to(torch.int32)
    bstart = torch.arange(nblocks, device=dev) * bm
    bexp = torch.searchsorted(pad_end, bstart, right=True)
    bexp = torch.where(bstart < pad_end[-1], bexp, -1).to(torch.int32)
    return routes, bexp, nblocks


def _one_chunk(x, topk_weights, topk_ids, lookups, tensors, alphas, bm, down, slice_channels):
    lib = _lib()
    T, H = x.shape
    top_k = topk_ids.shape[1]
    E = lookups.shape[1]
    cw1, cb1, cs1, hw1, hs1, hg1 = tensors[0:6]
    cw2, cb2, cs2, hw2, hs2, hg2 = tensors[8:14]
    N13 = hw1.shape[1] * 16
    I = N13 // 2
    dev = x.device
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    routes, bexp, nblocks = _route_blocks(topk_ids, E, bm)
    xs = x if x.dtype in (torch.bfloat16, torch.float16) else x.to(torch.bfloat16)
    xs = xs.contiguous()
    xp = torch.empty((T, H), dtype=torch.float16, device=dev)
    _check(lib.arvq_permute_x(_p(xs), _p(xp), T, H, int(xs.dtype == torch.bfloat16), stream),
           "arvq_permute_x")
    lk = lookups.contiguous()
    tw = topk_weights.reshape(-1)
    tw = tw if tw.dtype == torch.float32 and tw.is_contiguous() else tw.float().contiguous()
    h = torch.empty((nblocks * bm, I), dtype=torch.float16, device=dev)
    _check(lib.arvq_grouped_gateup(
        bm, _p(xp), _p(routes), _p(bexp), _p(lk), E, _p(cw1), _p(cb1), _p(cs1),
        _p(hw1), _p(hs1), _p(hg1), float(alphas[0]), _p(tw), _p(h), H, N13, top_k,
        hg1.shape[1] if hg1.ndim == 2 and hg1.shape[1] else 2, nblocks, stream, 0),
        "arvq_grouped_gateup")
    del xp
    if down == "atomic":
        out = torch.zeros((T, H), dtype=torch.float32, device=dev)
        fn, sub = lib.arvq_grouped_down, max(1, slice_channels // 128) if slice_channels else 0
    else:
        out = torch.empty((T * top_k, H), dtype=torch.bfloat16, device=dev)
        fn, sub = lib.arvq_grouped_down_store, 0
    _check(fn(
        bm, _p(h), _p(routes), _p(bexp), _p(lk), E, _p(cw2), _p(cb2), _p(cs2),
        _p(hw2), _p(hs2), _p(hg2), float(alphas[1]), _p(tw), _p(out), I, H, top_k,
        1, nblocks, stream, sub), "arvq_grouped_down")
    if down == "atomic":
        return out.to(x.dtype)
    res = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    _check(lib.arvq_route_sum(_p(out), _p(res), T, H, top_k, stream), "arvq_route_sum")
    return res if x.dtype == torch.bfloat16 else res.to(x.dtype)


def batched_prefill(x, topk_weights, topk_ids, lookups, tensors, alphas,
                    chunk_tokens=None, bm=32, down="atomic", slice_channels=512,
                    max_tokens=None):
    """Routed MLP output [T, hidden] (x.dtype). Temporaries per token chunk:
    FP16 x copy (2*hidden B/token), FP32 output (4*hidden B/token) and the FP16
    intermediate (~top_k*I*2 B/token): ~46 KB/token at TP4, 0.75 GB at 16K."""
    max_tokens = max_tokens or _CHUNK_TOKENS
    T = x.shape[0]
    if T <= max_tokens:
        return _one_chunk(x, topk_weights, topk_ids, lookups, tensors, alphas, bm, down,
                          slice_channels)
    return torch.cat([
        _one_chunk(x[s:s + max_tokens], topk_weights[s:s + max_tokens],
                   topk_ids[s:s + max_tokens], lookups, tensors, alphas, bm, down,
                   slice_channels)
        for s in range(0, T, max_tokens)
    ])


def batched_cold_prefill(x, topk_weights, topk_ids, lookups, tensors, alphas,
                         projection=None, chunk_tokens=128):
    """Signature-compatible with the existing grouped prefill entry point
    (``projection`` / ``chunk_tokens`` accepted and unused)."""
    del projection, chunk_tokens
    return batched_prefill(x, topk_weights, topk_ids, lookups, tensors, alphas)
