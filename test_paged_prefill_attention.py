"""
Paged prefill kernel（prefix cache）独立正确性测试 + 基准。

用法：
    python test_paged_prefill_attention.py              # 只跑正确性
    python test_paged_prefill_attention.py --benchmark  # 额外跑 benchmark（需要 flash_attn）
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
                kv_block_size=256, num_blocks=None, dtype=torch.bfloat16, seed=0):
    """构造 prefix cache 场景：每个 seq 有 cached = seqlen_k - seqlen_q 个缓存 token。"""
    assert all(k >= q for k, q in zip(seq_lens_k, seq_lens_q))
    torch.manual_seed(seed)
    B = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device="cuda")

    if num_blocks is None:
        num_blocks = sum((l + kv_block_size - 1) // kv_block_size for l in seq_lens_k)
    k_cache = torch.randn(num_blocks, kv_block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_cache = torch.randn(num_blocks, kv_block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")

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


def ref_paged_prefill(q, k_cache, v_cache, block_table, cu_q, cu_k, scale):
    """纯 PyTorch 参考实现，O(n^2)，只用于小用例对拍。"""
    q = q.float(); k_cache = k_cache.float(); v_cache = v_cache.float()
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_block_size = k_cache.shape[1]
    gqa = num_heads // num_kv_heads
    B = len(cu_q) - 1
    out = torch.empty_like(q)
    for s in range(B):
        q_start, q_end = int(cu_q[s]), int(cu_q[s + 1])
        seqlen_q = q_end - q_start
        seqlen_k = int(cu_k[s + 1] - cu_k[s])
        cached = seqlen_k - seqlen_q
        # 展开分页 K/V 成连续逻辑序列
        n_blocks = (seqlen_k + kv_block_size - 1) // kv_block_size
        ks, vs = [], []
        for b in range(n_blocks):
            bid = int(block_table[s, b].item())
            ks.append(k_cache[bid]); vs.append(v_cache[bid])
        k = torch.cat(ks, dim=0)[:seqlen_k]   # (seqlen_k, num_kv_heads, head_dim)
        v = torch.cat(vs, dim=0)[:seqlen_k]
        for j in range(q_start, q_end):
            abs_pos = cached + (j - q_start)   # 绝对位置
            for h in range(num_heads):
                kv_h = h // gqa
                scores = (q[j, h] @ k[:abs_pos + 1, kv_h].T) * scale
                p = torch.softmax(scores, dim=0)
                out[j, h] = p @ v[:abs_pos + 1, kv_h]
    return out


def run_test(name, seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim):
    q, k_cache, v_cache, block_table, cu_q, cu_k, scale = make_inputs(
        seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim)
    out = paged_prefill_attention(q, k_cache, v_cache, cu_q, cu_k, block_table, scale)
    ref = ref_paged_prefill(q, k_cache, v_cache, block_table, cu_q, cu_k, scale)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    print(f"[PASS] {name}")


def run_flash_test(name, seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim):
    q, k_cache, v_cache, block_table, cu_q, cu_k, scale = make_inputs(
        seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim)
    out = paged_prefill_attention(q, k_cache, v_cache, cu_q, cu_k, block_table, scale)
    fa = flash_attn_varlen_func(
        q, k_cache, v_cache,
        max_seqlen_q=max(seq_lens_q), cu_seqlens_q=cu_q,
        max_seqlen_k=max(seq_lens_k), cu_seqlens_k=cu_k,
        softmax_scale=scale, causal=True, block_table=block_table,
    )
    torch.testing.assert_close(out.float(), fa.float(), atol=1e-2, rtol=1e-2)
    print(f"[PASS] {name}")


def benchmark(seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim):
    q, k_cache, v_cache, block_table, cu_q, cu_k, scale = make_inputs(
        seq_lens_q, seq_lens_k, num_heads, num_kv_heads, head_dim)

    def run_triton():
        return paged_prefill_attention(q, k_cache, v_cache, cu_q, cu_k, block_table, scale)

    def run_flash():
        return flash_attn_varlen_func(
            q, k_cache, v_cache,
            max_seqlen_q=max(seq_lens_q), cu_seqlens_q=cu_q,
            max_seqlen_k=max(seq_lens_k), cu_seqlens_k=cu_k,
            softmax_scale=scale, causal=True, block_table=block_table,
        )

    ms_t = triton.testing.do_bench(run_triton)
    ms_f = triton.testing.do_bench(run_flash)
    print(f"  triton paged prefill: {ms_t * 1e3:.1f} us")
    print(f"  flash_attn (paged)   : {ms_f * 1e3:.1f} us")
    print(f"  占比                  : {ms_t / ms_f * 100:.1f}%")


if __name__ == "__main__":
    import sys

    # 正确性（vs 参考实现）：cached=0 退化成普通 prefill，cached>0 是真 prefix cache
    run_test("MHA cached=0 (head_dim 64)", [128], [128], 4, 4, 64)
    run_test("MHA cached>0 (head_dim 64)", [128], [256], 4, 4, 64)
    run_test("MHA 多 seq 混合 cached", [64, 128, 256], [64, 300, 511], 4, 4, 64)
    run_test("GQA 2:1 cached (head_dim 128)", [128, 257], [512, 511], 8, 4, 128)
    run_test("GQA 4:1 cached", [63, 513], [511, 1024], 16, 4, 128)

    print("\n--- 和 flash_attn_varlen_func (block_table) 对拍 ---")
    run_flash_test("flash MHA cached (head_dim 64)", [64, 128], [64, 300], 4, 4, 64)
    run_flash_test("flash GQA 2:1 cached", [128, 257], [512, 511], 8, 4, 128)
    run_flash_test("flash GQA 4:1 cached", [63, 513], [511, 1024], 16, 4, 128)

    if "--benchmark" in sys.argv:
        print("\n--- benchmark (32 seqs: cached 2048 + new 2048, GQA 2:1) ---")
        benchmark([2048] * 32, [4096] * 32, 16, 8, 128)

    print("\n全部 paged prefill 测试通过。")
