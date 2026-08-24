# M5 结果 —— Triton prefill kernel 替换 flash_attn_varlen_func

- 日期：2026-08-23
- 环境：RTX 4090 (sm_89)，flash_attn 2.8.3.post1，torch 2.6.0a0 (cu128)，triton 3.1.0
- 结果：✅ 正确性 6/6 通过，端到端跑通（Prefill 8 tok/s），benchmark 达 flash_attn 的 122.6%

## 结论

自研 FlashAttention-2 风格 Triton prefill kernel（varlen + causal + GQA + tensor core），替换掉 nano-vllm 普通 prefill 分支的 `flash_attn_varlen_func`：

| | 结果 |
|---|---|
| 正确性（vs PyTorch ref + flash_attn_varlen_func） | 6/6 通过 |
| benchmark（32×4096, GQA 2:1, head_dim 128） | 17.3ms vs flash 14.1ms = **122.6%** |
| 端到端（example.py, Qwen3-0.6B, head_dim 64） | Prefill **8 tok/s**（flash 版 7 tok/s），生成文本正确 |

最优配置：`BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3`。

---

## Debug 完整时间线（报错 → 定位 → 修复 → 验证）

### Bug 1：多序列 NaN —— online softmax 的 `-inf - -inf`

**报错（第一次跑测试）**：

```
$ python test_prefill_attention.py
[PASS] MHA single seq len 128
Traceback (most recent call last):
  File ".../test_prefill_attention.py", line 61, in run_test
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
AssertionError: Tensor-likes are not close!
Mismatched elements: 98304 / 114688 (85.7%)
Greatest absolute difference: nan at index (64, 0, 0)
```

三个关键线索：
1. 第一个**单序列**用例过了，**多序列**用例挂。
2. nan 从 index `(64, 0, 0)` 开始——`64` 正是第二个序列的第一个 token（第一个序列长 64）。
3. `98304 / 16384 = 6`：448 个 token 切成 7 个 block，只有 block 0 对，block 1~6 全 nan。

**定位第 1 步：排除 seq_id 计算**。写了 `dbg_seq.py`，把 kernel 里的 `seq_id` 和 `pos` 直接 store 出来，跟 `torch.searchsorted` 对拍：

```
sid equal : True
pos equal : True
```

→ 地址翻译（seq_id / pos）完全正确，不是这的锅。

> 踩的坑：`dbg_seq.py` 第一版用 `python - << 'PYEOF'` 从 stdin 跑，报
> `OSError: could not get source code`——因为 `@triton.jit` 要 `inspect.getsourcelines`
> 读真实文件源码，stdin 读不到。改成 `cat > dbg_seq.py` 写文件再跑才成。

**定位第 2 步：锁定 online softmax**。seq_id 对、block 0 对但 block 1+ 全 nan，问题只可能在 softmax 更新。多序列时，block 1（序列 2 的 query）的第一个 key block 属于序列 1，causal 掩码整块 false → `qk` 一整行都是 `-inf`：

```
m_new = max(m_i, row_max)      # m_i=-inf, row_max=-inf → m_new=-inf
alpha = exp(m_i - m_new)       # -inf - (-inf) = NaN
p     = exp(qk - m_new)        # 同样 NaN
```

单序列不会触发（第一个 key block 一定是对角线有效块），只有 varlen 多序列才会。

**修复**：`m_new` 仍是 `-inf` 时把 `alpha` 和 `p` 显式置 0：

```python
m_new = tl.maximum(m_i, tl.max(qk, axis=1))
alpha = tl.where(m_new > float("-inf"), tl.exp(m_i - m_new), 0.0)
p = tl.exp(qk - m_new[:, None])
p = tl.where(m_new[:, None] > float("-inf"), p, 0.0)
```

**验证**：`python test_prefill_attention.py` → 6/6 全过。

---

### Bug 2：6126% 性能灾难 —— O(B²) 跨序列扫描

**报错（第一次 benchmark）**：

```
--- benchmark (32 seqs x 4096, GQA 2:1) ---
  triton prefill: 822523.9 us
  flash_attn    : 13426.5 us
  占比           : 6126.1%
```

慢 61 倍。822ms 对比 flash 13ms，这是算法级缺陷，不是调参能救的。

**根因**：`num_n_blocks = tl.cdiv(hi, BLOCK_N)`，其中 `hi = start_m + BLOCK_M` 是**全局 token 上界**。多序列 varlen 时，序列 j 的 query block 会从 token 0 开始循环，把前面 j 个序列的 key block 全扫一遍——虽然它们全被 causal 掩码掉，纯浪费。32 个序列 = O(B²) ≈ 15 倍，叠上循环内 seq_id 计算的开销到 61 倍。

**修复**：循环下界从 0 改为「本 block 最早序列的起始位置」：

```python
start_block = tl.min(starts_m) // BLOCK_N
for start_n in tl.range(start_block, num_n_blocks):
```

跳过的 key 一定属于别的序列（必被 mask），数学结果不变，纯消除浪费。

**验证**：`29485.1 us vs 13393.2 = 220.1%`（快 28 倍），正确性保持。

---

### 调参：BLOCK / num_warps / num_stages 扫描

**第一轮 sweep**，BLOCK_N=128 和 BLOCK_M=256 全 OOM：

```
BLOCK_M=128 BLOCK_N=128 warps=8: FAILED OutOfResources: out of resource: shared memory,
  Required: 164352, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.
```

最好 `128/64/8 = 136.4%`。

**加 num_stages 参数**再扫：`num_stages=4` 也 OOM（shared memory 更紧张），`num_stages=2` 更慢（153.7%）。最好还是 136%。

→ 结论：卡在 136% 不是 tile 大小问题，是**循环内每轮重复做 seq_id 计算**。

---

### 重构：按序列分块（220% → 122.6%）

循环里每轮迭代都要做 `seq_id` 的 `[BLOCK_N, MAX_SEQS]` 广播比较 + `cu_seqlens` gather，这是 2.2 倍差距的来源。重构：

- grid 第 0 维从「全局 token block」改成「**序列内的 query block**」
- 每个 program 用 `cum_blocks`（每序列 block 计数前缀和）一次性定位 `(seq_id, block_in_seq)`
- 循环只在**序列内**迭代 key block，causal 退化成纯 `pos_n <= pos_m`，不再有 seq_id 比较和 gather

**验证**：`17300.9 vs 14107.1 = 122.6%`。

---

## 性能扫描最终记录（32×4096, head_dim 128）

| 配置 | us | 占比 |
|---|---|---|
| 128/64/8/3 | 17300.9 | **122.6%**（最优） |
| 128/64/8/2 | 18496.7 | 131.1% |
| 128/128/8/1 | 18779.9 | 133.1% |
| 128/64/4/4 | FAILED | OOM（shared memory） |
| 256/64/8 | FAILED | OOM（shared memory） |

---

## 和 decode 的本质区别（为什么一个有 parity 一个没有）

| | decode（M4） | prefill（M5） |
|---|---|---|
| 算术强度 | 0.5 FLOP/byte（memory-bound） | 高（compute-bound） |
| tensor core | 用不了（M=1 < 16） | 能用（M=BLOCK_M≥16） |
| 结果 | 打平 flash_attn（100.5%） | 慢 22.6%（122.6%） |

decode 打平是因为**两边都饱和带宽**；prefill 是 compute-bound，flash_attn 的对角线 block split、warp 调度、smem 管理这些深度优化，朴素 Triton 吃不到，慢 ~20% 是正常水位。

## 面试可讲点

- varlen 多序列的 causal mask：先 seq_id + pos，再退化成按序列分块的 `pos_n <= pos_m`
- online softmax 整块被 mask 时的 `-inf - -inf = NaN` 陷阱（单序列不会踩，只有 varlen 才踩）
- 61 倍 → 2.2 倍 → 1.23 倍的渐进优化路径（全局上界 → start_block → 按序列分块）
- compute-bound（prefill）vs memory-bound（decode）决定能否打平 flash_attn
- tensor core 在 prefill 能用（M≥16）、decode 不能用（M=1），是同一套约束的两面
- Triton `@jit` 要读真实文件源码，stdin 跑不了（`OSError: could not get source code`）
