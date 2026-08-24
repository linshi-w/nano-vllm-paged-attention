"""
Paged prefill kernel 配置扫描（BLOCK_M / BLOCK_N / num_warps / num_stages）。

用法：python sweep_paged_prefill.py
"""

import importlib.util
import pathlib

import torch
import triton
import triton.testing

_kernel_spec = importlib.util.spec_from_file_location(
    "paged_prefill_attention",
    pathlib.Path(__file__).resolve().parent / "nanovllm" / "layers" / "paged_prefill_attention.py",
)
_kernel = importlib.util.module_from_spec(_kernel_spec)
_kernel_spec.loader.exec_module(_kernel)
paged_prefill_attention = _kernel.paged_prefill_attention

from flash_attn import flash_attn_varlen_func


def make_inputs(seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim,
                kv_block_size=256, num_blocks=None, seed=0):
    torch.manual_seed(seed)
    B = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    if num_blocks is None:
        num_blocks = sum((l + kv_block_size - 1) // kv_block_size for l in seq_lens_k)
    k_cache = torch.randn(num_blocks, kv_block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(num_blocks, kv_block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    max_blocks = max((l + kv_block_size - 1) // kv_block_size for l in seq_lens_k)
    block_table = torch.full((B, max_blocks), -1, dtype=torch.int32, device="cuda")
    next_block = 0
    for s, l in enumerate(seq_lens_k):
        nb = (l + kv_block_size - 1) // kv_block_size
        block_table[s, :nb] = torch.arange(next_block, next_block + nb, dtype=torch.int32, device="cuda")
        next_block += nb
    cu_q = torch.tensor([0] + torch.cumsum(torch.tensor(seq_lens_q), 0).tolist(), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0] + torch.cumsum(torch.tensor(seq_lens_k), 0).tolist(), dtype=torch.int32, device="cuda")
    scale = head_dim ** -0.5
    return q, k_cache, v_cache, block_table, cu_q, cu_k, scale


if __name__ == "__main__":
    seq_lens_q = [2048] * 32
    seq_lens_k = [4096] * 32
    num_heads, num_kv_heads, head_dim = 16, 8, 128
    q, k_cache, v_cache, block_table, cu_q, cu_k, scale = make_inputs(
        seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim)

    def run_flash():
        return flash_attn_varlen_func(
            q, k_cache, v_cache,
            max_seqlen_q=max(seq_lens_q), cu_seqlens_q=cu_q,
            max_seqlen_k=max(seq_lens_k), cu_seqlens_k=cu_k,
            softmax_scale=scale, causal=True, block_table=block_table,
        )

    ms_f = triton.testing.do_bench(run_flash)
    print(f"flash_attn (paged): {ms_f * 1e3:.1f} us\n")

    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128]:
            for warps in [4, 8]:
                for stages in [2, 3]:
                    try:
                        def run_t():
                            return paged_prefill_attention(
                                q, k_cache, v_cache, cu_q, cu_k, block_table, scale,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                num_warps=warps, num_stages=stages)
                        ms_t = triton.testing.do_bench(run_t)
                        print(f"BLOCK_M={BLOCK_M:3d} BLOCK_N={BLOCK_N:3d} warps={warps} stages={stages}: "
                              f"{ms_t * 1e3:8.1f} us ({ms_t / ms_f * 100:5.1f}%)")
                    except Exception as e:
                        print(f"BLOCK_M={BLOCK_M:3d} BLOCK_N={BLOCK_N:3d} warps={warps} stages={stages}: FAILED {e}")
