"""
PagedAttention Triton kernel 的独立正确性测试 + 基准。

用法：
    python test_paged_attention.py              # 只跑正确性测试
    python test_paged_attention.py --benchmark  # 额外跑 benchmark（需要 flash_attn）

正确性测试覆盖：
    - MHA（num_heads == num_kv_heads）和 GQA（num_heads > num_kv_heads）
    - seq_len 不是 block_size 整数倍（触发最后一个不完整 block）
    - 多个不同长度的 seq 打包
"""

import importlib.util
import math
import pathlib

import torch
import triton
import triton.testing

# 直接按文件路径加载 paged_attention.py，避免触发 nanovllm/__init__.py
# （那会拽进 xxhash/transformers/flash_attn 等一堆无关依赖）。
_pa_spec = importlib.util.spec_from_file_location(
    "paged_attention",
    pathlib.Path(__file__).resolve().parent / "nanovllm" / "layers" / "paged_attention.py",
)
_pa = importlib.util.module_from_spec(_pa_spec)
_pa_spec.loader.exec_module(_pa)
paged_attention = _pa.paged_attention

try:
    from flash_attn import flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def ref_paged_attention(q, k_cache, v_cache, block_table, seq_lens, scale):
    """纯 PyTorch 参考实现，fp32 累加，用于对拍正确性。"""
    q = q.float()
    k_cache = k_cache.float()
    v_cache = v_cache.float()
    num_seqs, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_block_size = k_cache.shape[1]
    num_q_per_kv = num_heads // num_kv_heads

    out = torch.empty_like(q)
    for s in range(num_seqs):
        seq_len = int(seq_lens[s].item())
        n_blocks = (seq_len + kv_block_size - 1) // kv_block_size
        ks, vs = [], []
        for b in range(n_blocks):
            bid = int(block_table[s, b].item())
            ks.append(k_cache[bid])
            vs.append(v_cache[bid])
        k = torch.cat(ks, dim=0)[:seq_len]   # (seq_len, num_kv_heads, head_dim)
        v = torch.cat(vs, dim=0)[:seq_len]
        for h in range(num_heads):
            kv_h = h // num_q_per_kv
            scores = (q[s, h] @ k[:, kv_h].T) * scale
            p = torch.softmax(scores, dim=0)
            out[s, h] = p @ v[:, kv_h]
    return out


def make_inputs(num_seqs, num_heads, num_kv_heads, head_dim, kv_block_size,
                seq_lens, num_blocks, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device="cuda")
    k_cache = torch.randn(num_blocks, kv_block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_cache = torch.randn(num_blocks, kv_block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    max_blocks = max((l + kv_block_size - 1) // kv_block_size for l in seq_lens)
    # 每个 seq 分配它需要的 block（顺序分配，保证 block_id < num_blocks）
    block_table = torch.full((num_seqs, max_blocks), -1, dtype=torch.int32, device="cuda")
    next_block = 0
    for s, l in enumerate(seq_lens):
        nb = (l + kv_block_size - 1) // kv_block_size
        ids = torch.arange(next_block, next_block + nb, dtype=torch.int32, device="cuda")
        block_table[s, :nb] = ids
        next_block += nb

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    scale = head_dim ** -0.5
    return q, k_cache, v_cache, block_table, seq_lens_t, scale


def run_test(name, num_seqs, num_heads, num_kv_heads, head_dim, kv_block_size, seq_lens):
    num_blocks = 64
    q, k_cache, v_cache, block_table, seq_lens_t, scale = make_inputs(
        num_seqs, num_heads, num_kv_heads, head_dim, kv_block_size, seq_lens, num_blocks)

    out = paged_attention(q, k_cache, v_cache, block_table, seq_lens_t, scale)
    ref = ref_paged_attention(q, k_cache, v_cache, block_table, seq_lens_t, scale)

    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    print(f"[PASS] {name}")


def run_flash_attn_test(name, num_seqs, num_heads, num_kv_heads, head_dim, kv_block_size, seq_lens):
    if not HAS_FLASH_ATTN:
        print(f"[SKIP] {name} (flash_attn 未安装)")
        return
    q, k_cache, v_cache, block_table, seq_lens_t, scale = make_inputs(
        num_seqs, num_heads, num_kv_heads, head_dim, kv_block_size, seq_lens, 64)

    out = paged_attention(q, k_cache, v_cache, block_table, seq_lens_t, scale)
    fa = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=seq_lens_t, block_table=block_table,
        softmax_scale=scale, causal=True,
    ).squeeze(1)

    torch.testing.assert_close(out.float(), fa.float(), atol=1e-2, rtol=1e-2)
    print(f"[PASS] {name}")


def benchmark(num_seqs=32, num_heads=16, num_kv_heads=8, head_dim=128,
              kv_block_size=256, seq_len=4096, num_blocks=512):
    if not HAS_FLASH_ATTN:
        print("[SKIP] benchmark 需要 flash_attn")
        return
    seq_lens = [seq_len] * num_seqs
    q, k_cache, v_cache, block_table, seq_lens_t, scale = make_inputs(
        num_seqs, num_heads, num_kv_heads, head_dim, kv_block_size, seq_lens, num_blocks)

    def run_triton():
        return paged_attention(q, k_cache, v_cache, block_table, seq_lens_t, scale)

    def run_flash():
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=seq_lens_t, block_table=block_table,
            softmax_scale=scale, causal=True,
        ).squeeze(1)

    ms_triton = triton.testing.do_bench(run_triton)
    ms_flash = triton.testing.do_bench(run_flash)
    print(f"\n--- benchmark (num_seqs={num_seqs}, seq_len={seq_len}, head_dim={head_dim}) ---")
    print(f"  triton paged_attention: {ms_triton * 1e3:.1f} us")
    print(f"  flash_attn (decode)   : {ms_flash * 1e3:.1f} us")
    print(f"  flash_attn 占比        : {ms_triton / ms_flash * 100:.1f}%")


if __name__ == "__main__":
    import sys

    # 正确性测试
    run_test("MHA 完整 block (head_dim=64)",  2, 4, 4, 64, 256, [256, 512])
    run_test("MHA 部分 block (head_dim=64)",  3, 4, 4, 64, 256, [100, 300, 600])
    run_test("GQA 2:1 (head_dim=128)",        4, 8, 4, 128, 256, [128, 257, 1000, 2048])
    run_test("GQA 4:1 (head_dim=128)",        2, 16, 4, 128, 256, [63, 513])
    run_test("MHA 小 block_size=64",          2, 4, 4, 128, 64, [64, 130])

    if HAS_FLASH_ATTN:
        print("\n--- 和 flash_attn_with_kvcache 对拍 ---")
        run_flash_attn_test("flash_attn GQA 2:1", 4, 8, 4, 128, 256, [128, 257, 1000])
        run_flash_attn_test("flash_attn MHA",     2, 8, 8, 128, 256, [256, 777])
    else:
        print("\n(flash_attn 未安装，跳过和它的对拍)")

    if "--benchmark" in sys.argv:
        benchmark()

    print("\n全部正确性测试通过。")
