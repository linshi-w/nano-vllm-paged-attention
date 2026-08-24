# M3 验证结果 —— benchmark 对比 flash_attn

- 日期：2026-08-15
- 环境：RTX 5090 (sm_120)，flash_attn 2.8.3.post1，torch 2.6.0a0 (cu128)
- 结果：✅ 全部通过

## 正确性

- 5/5 对拍纯 PyTorch 参考实现（MHA 完整/部分 block、GQA 2:1、GQA 4:1、小 block_size）
- 2/2 对拍 `flash_attn_with_kvcache`（输出一致：GQA 2:1、MHA）

## 性能 benchmark

配置：`num_seqs=32, seq_len=4096, head_dim=128, GQA 2:1`（num_heads=16, num_kv_heads=8）

| kernel | 耗时 |
|---|---|
| triton paged_attention | 404.3 us |
| flash_attn_with_kvcache | 359.1 us |

自研 kernel = flash_attn 的 **112.6%** 耗时（慢 12.6%）。

## 解读（面试可讲）

decode 是 memory-bound：每个 seq 每步只算 1 个 query token，却要从 KV cache 读 `seq_len`（4096）个 token 的 K/V。瓶颈是显存带宽（读 KV），不是计算。因此从零手写的 Triton kernel 能追平手写 CUDA 的 flash_attn——差距仅 12.6%，几乎打平。

## 备注

- 修复了 `test_paged_attention.py` 的一个 bug：第一个 flash_attn 对拍写 `num_seqs=4` 但 seq 长度列表只有 3 个元素（漏 `2048`），导致 `seqlens_k must have shape (batch_size)`。本地已改为 `[128, 257, 1000, 2048]`。
- M3 完成。M4（替换 prefill 的 `flash_attn_varlen_func`，彻底去 flash_attn 依赖）为可选加分项，寒假再做。
