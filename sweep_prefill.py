"""prefill kernel BLOCK_M/BLOCK_N/num_warps/num_stages 扫描。"""
import importlib.util, pathlib

import torch, triton, triton.testing

_pa_spec = importlib.util.spec_from_file_location(
    "prefill_attention", pathlib.Path("nanovllm/layers/prefill_attention.py"))
_pa = importlib.util.module_from_spec(_pa_spec)
_pa_spec.loader.exec_module(_pa)
prefill_attention = _pa.prefill_attention

from flash_attn import flash_attn_varlen_func

seq_lens = [4096] * 32
num_heads, num_kv_heads, head_dim = 16, 8, 128

torch.manual_seed(0)
total = sum(seq_lens)
q = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
k = torch.randn(total, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
v = torch.randn(total, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
cum = [0] + torch.cumsum(torch.tensor(seq_lens), 0).tolist()
cu = torch.tensor(cum, dtype=torch.int32, device="cuda")
scale = head_dim ** -0.5
max_len = max(seq_lens)


def run_flash():
    return flash_attn_varlen_func(q, k, v, max_seqlen_q=max_len, cu_seqlens_q=cu,
                                  max_seqlen_k=max_len, cu_seqlens_k=cu,
                                  softmax_scale=scale, causal=True)


ms_f = triton.testing.do_bench(run_flash)
print(f"flash_attn: {ms_f * 1e3:8.1f} us")

configs = [
    (128, 64, 8, 3),
    (128, 64, 8, 4),
    (128, 64, 8, 2),
    (128, 64, 4, 4),
    (128, 128, 8, 1),
    (128, 128, 8, 2),
    (64, 64, 4, 4),
    (64, 128, 4, 1),
    (64, 64, 4, 2),
]

for bm, bn, nw, ns in configs:
    def run_triton():
        return prefill_attention(q, k, v, cu, scale, BLOCK_M=bm, BLOCK_N=bn,
                                 num_warps=nw, num_stages=ns)
    try:
        ms_t = triton.testing.do_bench(run_triton)
        print(f"BLOCK_M={bm:3d} BLOCK_N={bn:3d} warps={nw} stages={ns}: {ms_t * 1e3:8.1f} us  ({ms_t / ms_f * 100:.1f}%)")
    except Exception as e:
        print(f"BLOCK_M={bm:3d} BLOCK_N={bn:3d} warps={nw} stages={ns}: FAILED {type(e).__name__}")
