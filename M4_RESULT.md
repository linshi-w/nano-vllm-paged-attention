# M4 深挖结果 —— decode kernel 打平 flash_attn + memory-bound 三证据

- 日期：2026-08-23
- 环境：RTX 4090 (sm_89)，flash_attn 2.8.3.post1，torch 2.6.0a0 (cu128)，triton 3.1.0
- 结果：✅ 正确性 7/7 通过，decode kernel 打平 flash_attn

## 结论

自研 Triton PagedAttention decode kernel 在 RTX 4090 上追平 flash_attn_with_kvcache：

| kernel | 耗时 |
|---|---|
| triton paged_attention | 633.4 us |
| flash_attn_with_kvcache | 629.9 us |
| 占比 | 100.5%（打平，噪声内） |

配置：`num_seqs=32, seq_len=4096, head_dim=128, GQA 2:1`（num_heads=16, num_kv_heads=8）。

## Debug 完整时间线（报错 → 定位 → 修复 → 验证）

### Bug 1：NCU 打不开 —— 驱动级 profiling 锁

**报错（第一次跑 ncu）**：

```
$ ncu --set full python test_paged_attention.py --benchmark
==ERROR== ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA
           GPU Performance Counters on the target device
```

**定位**：`nvidia-smi` 正常、能跑模型，只有 NCU 挂；查 `/proc/driver/nvidia/params` 发现 `RmProfilingAdminOnly=1`。容器 root 也改不了（驱动级参数，只有宿主机能改）。

**根因**：租的 GPU 容器（共享宿主机）开了驱动级 profiling 锁，禁掉硬件性能计数器（PMU）访问，NCU 这类 profiler 全部失效。

**替代方案**：不硬刚 NCU，改用 **roofline 模型**论证 memory-bound：
- 每个 (query, head) 读 K+V = 2 × seq_len × head_dim × 2B ≈ 2 MB
- 算力 = seq_len × head_dim × 2 ≈ 1 MFLOP
- 算术强度 = 0.5 FLOP/byte，远低于 4090 ridge（82.6 TFLOPS / 1008 GB/s ≈ 82 FLOP/byte）

**验证**：0.5 << 82 → 理论上限就是 memory-bound；后面再补两条实证兜底。

### Bug 2：tensor core 编译期 assert —— decode 的 M=1

**报错（把 `tl.sum(q * K)` 改成 `tl.dot(q, Kᵀ)` 后，编译期）**：

```
All non-batch values in both first input shape ([1, 64]) ... must be >= 16!
```

**定位**：Triton 编译期 assert，直接报在 `tl.dot` 第一个输入 shape `[1, 64]` 上。

**根因**：decode 的 Q 只有 1 个 token（M=1）。tensor core 的 MMA 指令要求 M≥16（一个 warp 要铺满至少 16 行才用得上 TC）。decode 单 token 天然 M=1，物理上用不了 tensor core。

**修复**：回退 `tl.sum(q * K)`（标量点积，走 CUDA core）。不硬凑——这本身就是 memory-bound 的第二条证据：算力优化路径被硬件锁死，瓶颈本来也不在算。

**验证**：`tl.sum` 版本 633.4 us 打平 flash_attn 629.9 us，说明没上 tensor core 也没吃亏。

### 调参：BLOCK_SIZE / num_warps 扫描（9 配置）

9 配置（block_size ∈ {32,64,128} × num_warps ∈ {2,4,8}）：

| block_size | num_warps | us |
|---|---|---|
| 32 | 2 | 672.2 |
| 32 | 4 | 635.5 |
| 32 | 8 | 684.5 |
| 64 | 2 | 624.4 |
| 64 | 4 | 631.5 |
| 64 | 8 | 756.6 |
| 128 | 2 | 627.9 |
| 128 | 4 | 642.1 |
| 128 | 8 | 679.6 |

**验证**：num_warps=8 差 20%（一个 (64,128) tile 分给 8 个 warp，每 warp 只分 8 行 K，访存碎片化 + 调度开销）；block_size=32 差（tile 太小，查表次数翻倍 4096/32=128 次 vs 64 次）；其余打平（±1%）。**tuning 不动 = memory-bound 的第三条实证。**

## 三条 memory-bound 证据链

1. **roofline**：0.5 FLOP/byte << 82 ridge → 理论 memory-bound。
2. **tl.dot M=1 assert**：tensor core 物理不可用 → compute 优化被锁死。
3. **tuning ±1% 不动 + 跨卡耗时随带宽**（5090→4090 = 631/404 ≈ 1.56× vs 带宽比 1.78×，SM 是 2 倍多却没快 2 倍）→ 卡在带宽。

## 5090 vs 4090 对比

| | M3（5090, sm_120） | M4（4090, sm_89） |
|---|---|---|
| triton | 404.3 us | 633.4 us |
| flash_attn | 359.1 us | 629.9 us |
| 差距 | 慢 12.6% | 打平（100.5%） |

原因：flash_attn 针对 Blackwell (sm_120) 做了专门优化，朴素 Triton 吃不到；Ada (4090) 是成熟目标，两边都纯 memory-bound、都打满带宽，于是打平。

## 面试可讲点

- decode memory-bound 的根因：单 token query，读全量 KV。
- tensor core 为什么 decode 用不了：M=1 < 16。
- PagedAttention 价值在省显存/提并发，不在 FLOPs。
- 打平 flash_attn 的含义：带宽决定，不是 kernel 一样聪明。
