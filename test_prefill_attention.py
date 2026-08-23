"""
Prefill kernel 独立正确性测试 + 基准。

用法：
    python test_prefill_attention.py              # 只跑正确性
    python test_prefill_attention.py --benchmark  # 额外跑 benchmark（需要 flash_attn）
"""

import importlib.util
import pathlib

import torch
import triton
import triton.testing

_pa_spec = importlib.util.spec_from_file_location(
    "prefill_attention",
    pathlib.Path(__file__).resolve().parent / "nanovllm" / "layers" / "prefill_attention.py",
)
_pa = importlib.util.module_from_spec(_pa_spec)
_pa_spec.loader.exec_module(_pa)
prefill_attention = _pa.prefill_attention

from flash_attn import flash_attn_varlen_func


def make_inputs(seq_lens, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    total = sum(seq_lens)
    q = torch.randn(total, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    cum = [0] + torch.cumsum(torch.tensor(seq_lens), 0).tolist()
    cu = torch.tensor(cum, dtype=torch.int32, device="cuda")
    scale = head_dim ** -0.5
    return q, k, v, cu, scale


def ref_prefill(q, k, v, cu, scale):
    """纯 PyTorch 参考实现，O(n^2)，只用于小用例对拍。"""
    q = q.float(); k = k.float(); v = v.float()
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    gqa = num_heads // num_kv_heads
    out = torch.empty_like(q)
    for s in range(len(cu) - 1):
        start, end = int(cu[s]), int(cu[s + 1])
        for m in range(start, end):
            for h in range(num_heads):
                kv_h = h // gqa
                scores = (q[m, h] @ k[start:m + 1, kv_h].T) * scale
                p = torch.softmax(scores, dim=0)
                out[m, h] = p @ v[start:m + 1, kv_h]
    return out


def run_test(name, seq_lens, num_heads, num_kv_heads, head_dim):
    q, k, v, cu, scale = make_inputs(seq_lens, num_heads, num_kv_heads, head_dim)
    out = prefill_attention(q, k, v, cu, scale)
    ref = ref_prefill(q, k, v, cu, scale)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    print(f"[PASS] {name}")


def run_flash_test(name, seq_lens, num_heads, num_kv_heads, head_dim):
    q, k, v, cu, scale = make_inputs(seq_lens, num_heads, num_kv_heads, head_dim)
    max_len = max(seq_lens)
    out = prefill_attention(q, k, v, cu, scale)
    fa = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=max_len, cu_seqlens_q=cu,
        max_seqlen_k=max_len, cu_seqlens_k=cu,
        softmax_scale=scale, causal=True,
    )
    torch.testing.assert_close(out.float(), fa.float(), atol=1e-2, rtol=1e-2)
    print(f"[PASS] {name}")


def benchmark(seq_lens, num_heads, num_kv_heads, head_dim):
    q, k, v, cu, scale = make_inputs(seq_lens, num_heads, num_kv_heads, head_dim)
    max_len = max(seq_lens)

    def run_triton():
        return prefill_attention(q, k, v, cu, scale)

    def run_flash():
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max_len, cu_seqlens_q=cu,
            max_seqlen_k=max_len, cu_seqlens_k=cu,
            softmax_scale=scale, causal=True,
        )

    ms_t = triton.testing.do_bench(run_triton)
    ms_f = triton.testing.do_bench(run_flash)
    print(f"  triton prefill: {ms_t * 1e3:.1f} us")
    print(f"  flash_attn    : {ms_f * 1e3:.1f} us")
    print(f"  占比           : {ms_t / ms_f * 100:.1f}%")


if __name__ == "__main__":
    import sys

    run_test("MHA single seq len 128", [128], 8, 8, 128)
    run_test("MHA multi seq (head_dim 64)", [64, 128, 256], 4, 4, 64)
    run_test("GQA 2:1 multi seq", [128, 257, 300], 8, 4, 128)
    run_test("GQA 4:1 len 511 + 63", [511, 63], 16, 4, 128)

    print("\n--- 和 flash_attn_varlen_func 对拍 ---")
    run_flash_test("flash MHA (head_dim 64)", [64, 128, 256], 4, 4, 64)
    run_flash_test("flash GQA 2:1", [128, 257, 300], 8, 4, 128)

    if "--benchmark" in sys.argv:
        print("\n--- benchmark (32 seqs x 4096, GQA 2:1) ---")
        benchmark([4096] * 32, 16, 8, 128)

    print("\n全部 prefill 测试通过。")
