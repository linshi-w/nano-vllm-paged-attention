# nano-vllm 项目完整总结（M1 → M3）

> 项目：手写 Triton PagedAttention kernel，替换 nano-vllm 推理引擎 decode 阶段的 flash_attn 调用。
> 状态：M1 ✅ / M2 ✅ / M3 ✅ 全部完成。M4（可选）未做。
> 日期：2026-08-15

---

## 一、一句话定位

> **"我把一个 vLLM 式推理引擎的 decode attention，从 flash_attn 库换成了自己手写的 Triton PagedAttention kernel——正确性对拍全过，性能只慢 flash_attn 12.6%。"**

这是简历旗舰项目的完整故事线。

---

## 二、背景与目标

**nano-vllm** 是一个开源的小型 LLM 推理引擎（vLLM 的简化版，约 1200 行 Python，作者 GeeeekExplorer）。它的 attention 计算原来直接调 `flash_attn` 库：

- prefill 阶段 → `flash_attn_varlen_func`
- decode 阶段 → `flash_attn_with_kvcache`

**目标**：把 decode 阶段的 `flash_attn_with_kvcache` 换成自己手写的 Triton kernel，作为 AI Infra 实习面试的简历项目。

**为什么选 decode 阶段**（而不是 prefill）：
1. decode 结构简单——每个 sequence 每步只生成 1 个新 token，Q 只有一行。
2. PagedAttention 是 AI Infra 面试头号考点。
3. 能端到端跑通 + benchmark 出真实数据（可辩护的数字）。

---

## 三、项目整体结构（两层架构）

```
nanovllm/
├── engine/    ← CPU 侧：调度与编排（"大脑"）
│   ├── llm_engine.py      # 主控：generate 的 while 循环
│   ├── scheduler.py       # 调度：选 seq、算 token、分 block
│   ├── block_manager.py   # 分页内存管理：分配/释放物理 block、xxHash 前缀缓存
│   ├── model_runner.py    # seq→GPU tensor、跑模型、采样、CUDA Graph
│   └── sequence.py        # 单个请求的状态（token、block_table、prefill/decode 标志）
│
├── layers/    ← GPU 侧：算子（"肌肉"，★ 改动在这里）
│   ├── attention.py       # ★ 核心：attention 计算 + KV cache 读写
│   ├── paged_attention.py # ★ 你写的 Triton kernel（本项目核心成果）
│   ├── layernorm.py       # RMSNorm
│   ├── linear.py          # 线性层（含张量并行）
│   ├── rotary_embedding.py# RoPE
│   ├── activation.py      # SiLU（SwiGLU）
│   ├── embed_head.py      # Embedding + LM Head
│   └── sampler.py         # 采样
│
├── models/qwen3.py        # Qwen3 结构
└── utils/
    ├── context.py         # 全局单例：跨层传 is_prefill/slot_mapping/block_table
    └── loader.py          # 权重加载
```

**关键点**：你的改动只在 GPU 侧（`attention.py` + 新增 `paged_attention.py`），CPU 侧调度逻辑一行没动。

---

## 四、一次推理的完整数据流（上下游核心）

```
llm.generate(prompts)
  → LLMEngine.generate()
      add_request：把每个 prompt 变成 Sequence 进队列
  → while 没生成完: step()          ← 每 step 一轮
      step 内部三步：
      ① scheduler.schedule()
           - 决定哪些 seq 这轮跑、每个算几个 token
           - block_manager 分配物理 block → 填进每个 seq 的 block_table
           - 算出 slot_mapping（每个 token 要写进 KV 池的哪个格子）
      ② model_runner.run()
           - seq 状态 → GPU tensor（input_ids、block_table、slot_mapping...）
           - set_context()：塞进全局 context 单例
           - model 前向（Qwen3 → 每层 attention + MLP）
           - 采样出下一个 token
      ③ scheduler.postprocess()
           - 更新 seq 状态、检查是否 EOS、释放用完的 block
```

**attention 前向链（你的 kernel 就嵌在这一步）**：

```
Qwen3ForCausalLM
  → embed_tokens
  → for 每一层 DecoderLayer:
        input_layernorm
        Qwen3Attention
          └─ layers/attention.py  Attention.forward(q,k,v)
               ├─ store_kvcache(k,v,slot_mapping)      # 写：当前 token 的 K/V 存进 KV 池
               ├─ if prefill: flash_attn_varlen_func   # 读：整段 prompt（仍用 flash_attn）
               └─ if decode:  paged_attention(...)     # ★ 读：你的 kernel（替换 flash_attn_with_kvcache）
        post_attention_layernorm
        Qwen3MLP
  → final_norm → sampler
```

---

## 五、五大核心概念（必须吃透）

### 1. KV cache 为什么要「分页」（PagedAttention）
- 普通 KV cache 每个 seq 连续预分配 `max_seq_len` 那么长 → 用不满就浪费显存、多请求长短不一就碎片。
- PagedAttention 借 OS 虚拟内存分页思想：KV cache 切成固定大小物理块（`kv_block_size=256`，一块装 256 个 token），按需分配。每个 seq 拿一张表 `block_table[seq] = [物理块3, 物理块17, 物理块42, -1, ...]`（`-1` 未使用）。
- **一句话：逻辑上连续，物理上离散，中间靠 block_table 翻译。**

### 2. 地址翻译（写 vs 读，是同一套翻译的两条路）
读第 `pos` 个 token 的 KV：
```
① block_idx       = pos // kv_block_size      # 第几个逻辑块
② offset_in_block = pos % kv_block_size      # 块内偏移
③ block_id        = block_table[seq][block_idx]   # 查表：逻辑块 → 物理块
④ 物理地址 = block_id * stride_block + offset_in_block * stride_token + kv_head * stride_head
```
三个 stride（KV 池布局 `(num_blocks, kv_block_size, num_kv_heads, head_dim)`）：
| 名字 | 值 | 含义 |
|---|---|---|
| `stride_block` | `kv_block_size × num_kv_heads × head_dim` | 跨一个物理块 |
| `stride_token` | `num_kv_heads × head_dim` | 跨一个 token 位置 |
| `stride_head` | `head_dim` | 跨一个 head |

**写与读的区别**：
- 写（`store_kvcache`）：scheduler 提前算好每个 token 的 `slot_mapping`（物理下标），直接 store。
- 读（`paged_attention`）：历史 slot 没持久化，只剩 block_table，现场用「pos + 查表」还原物理地址。

### 3. `context` 单例是跨层传参的桥梁
- CPU 侧 `model_runner` 每轮 `set_context()` 填入 `is_prefill`、`slot_mapping`、`block_table`、`context_lens`。
- GPU 侧 `attention.forward` 里 `get_context()` 取出来。
- 你的 kernel 需要的 `block_tables`、`context_lens` 就是这么来的。

### 4. online softmax（kernel 的核心算法）
维护三个 running 状态，逐 tile 合并，最终 `output = acc / l_i`：
```
m_new  = max(m_i, max(s))
alpha  = exp(m_i - m_new)      # 修正因子：旧结果对齐到新 max
p      = exp(s - m_new)
l_i    = l_i * alpha + sum(p)           # exp 累加和
acc    = acc * alpha + sum(p * V)       # 加权 V 累加
m_i    = m_new
```
- 全程不需要 `[seq_len, seq_len]` 的 score 矩阵，省显存，数学上严格等价一次性 softmax。
- `alpha = exp(m_i - m_new)`：新 tile 出现更大 max 时把旧的 l_i、acc 缩放到新量纲；max 没变大则 `alpha=1`，旧结果不动。两种情况统一成一个公式。

### 5. GQA 映射
query head 比 KV head 多，多个 query head 共用一个 KV head：
```
num_q_per_kv = num_heads // num_kv_heads    # 如 16 // 4 = 4（4:1）
kv_head = head_idx // num_q_per_kv
```
`head_idx: 0 1 2 3 | 4 5 6 7` → `kv_head: 0 0 0 0 | 1 1 1 1`。这个除法在 kernel 开头算好，全程复用。

---

## 六、M1 详解：独立 kernel + 正确性测试

### kernel 设计（`paged_attention.py`，127 行）
- `grid = (num_seqs, num_heads)`：每个 program 负责一个 `(seq, query_head)`，输出一个 head_dim 向量。
- **decode 特化**：Q 只有 1 个 token（每步新生成的），读一整行 `(head_dim,)`；KV 是全部历史，按 tile 循环读。
- **`BLOCK_SIZE=64` 整除 `kv_block_size=256`**：每个 tile 完整落在同一物理块，**只查一次表**，不会跨块。若跨块就得 tile 内每个 token 分别查表，慢很多。
- `bf16` 输入 / `fp32` 累加：`tl.load(...).to(tl.float32)`，保证点积数值稳定。

### 正确性测试（`test_paged_attention.py`）
- **关键技巧**：用 `importlib.util.spec_from_file_location` 直接按路径加载 `paged_attention.py`，**绕过 `nanovllm/__init__.py`**（否则会拽进 xxhash/transformers/flash_attn 一堆依赖）。
- 对拍纯 PyTorch 参考实现，覆盖：MHA 完整/部分 block、GQA 2:1、GQA 4:1、小 block_size=64。
- **结果：5/5 PASS**（RTX 5090）。

---

## 七、M2 详解：接入 engine 端到端跑通

只改 `attention.py` 的 decode 分支（一处调用）：

```python
# 改前
o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                            softmax_scale=self.scale, causal=True)
# 改后
o = paged_attention(q, k_cache, v_cache,
                    context.block_tables, context.context_lens, self.scale)
```

- decode 时 q 形状 `(num_seqs, num_heads, head_dim)`，正好是 `paged_attention` 期望的输入。
- **结果**：`example.py` 端到端跑通，生成两段文本。**Prefill 7 tok/s、Decode 61 tok/s**（未调优）。

---

## 八、M3 详解：benchmark 对比 flash_attn

### 测法
`triton.testing.do_bench` 对拍两个函数，配置 `num_seqs=32, seq_len=4096, head_dim=128, GQA 2:1`。

### 结果
| kernel | 耗时 |
|---|---|
| triton paged_attention | 404.3 us |
| flash_attn_with_kvcache | 359.1 us |

**你的 kernel = flash_attn 的 112.6% 耗时（慢 12.6%）。**

正确性额外对拍：**2/2 和 flash_attn 输出一致**（GQA 2:1、MHA）。

### 为什么能追平（面试必讲的 memory-bound）
decode 是 **memory-bound**：每个 seq 每步只算 1 个 query token，却要从 KV cache 读 `seq_len`（4096）个 token 的 K/V。**瓶颈是显存带宽（读 KV），不是计算**。当大家都卡在带宽上时，Triton 和手写 CUDA 的算力差距被拉平，所以差距只有 12.6%。

---

## 九、成果总表

| 里程碑 | 内容 | 结果 |
|---|---|---|
| M1 | Triton PagedAttention kernel + 单测 | 5/5 PASS（对拍 PyTorch） |
| M2 | 接入 engine decode 分支 | 端到端跑通，Prefill 7 / Decode 61 tok/s |
| M3 | benchmark 对比 flash_attn | 慢 12.6%，2/2 输出一致 |
| M4 | 替换 prefill（可选） | 未做，寒假 |

**GitHub**：`github.com/linshi-w/nano-vllm-paged-attention`（public），README 英文、含 benchmark 数字。

---

## 十、面试可展开的点

1. **memory-bound vs compute-bound**：为什么 decode 的 Triton kernel 能追平 CUDA 的 flash_attn。
2. **PagedAttention 地址翻译**：block_table 查表 → 物理地址，slot 概念。
3. **online softmax**：M/L/acc 三状态，为什么不需要 score 矩阵，`alpha = exp(m_i - m_new)` 的数学含义。
4. **`BLOCK_SIZE` 为何必须整除 `kv_block_size`**：保证 tile 不跨物理块，只查一次表。
5. **fp32 累加**：bf16 输入为什么 dot 要 fp32 accumulate（精度/数值稳定性）。
6. **`s = tl.where(valid, s, -inf)`**：最后一个不完整 tile 越界位置设 -inf，`exp(-inf)=0` 等价于不存在。
7. **和 flash_attn 差距可能在哪**：block_size、num_warps、occupancy、是否用 tensor core。

---

## 十一、M1–M3 Debug 时间线（报错 → 定位 → 根因 → 修复 → 验证）

> M1–M3 的 debug 集中在**环境/依赖/测试**三块，不是 kernel 算法 bug——kernel 本身的 online softmax、地址翻译第一次就写对了（照着 FlashAttention 公式 + 逐行推演写的）。真正的算法级 debug 在 M4（tensor core M=1）和 M5（prefill 的 NaN、O(B²)、分块重构），见各自 RESULT.md。

### Bug 1：import 测试文件时拽进一整个引擎的依赖

**报错**：`test_paged_attention.py` 里 `from nanovllm.layers.paged_attention import paged_attention` 一 import 就挂——`nanovllm/__init__.py` 会连带 import `xxhash` / `transformers` / `flash_attn` 等一整套引擎依赖，测试环境没装全就崩。

**定位**：`nanovllm/__init__.py` 顶层就 import 了引擎的 model/engine 模块，import 链把整套推理栈都拉起来了。

**根因**：测试只想 load 一个 127 行的 kernel 文件，但 Python 的包 import 会先执行 `__init__.py`。

**修复**：`importlib.util.spec_from_file_location` 按文件路径直接加载 `paged_attention.py`，绕过 `__init__.py`。

**验证**：单测可以脱离引擎依赖独立跑（只装 torch + triton 就行）。

### Bug 2：flash_attn 老版本没有 block_table 参数

**报错**：decode 对拍调用 `flash_attn_with_kvcache(..., block_table=...)` 直接 `TypeError`，提示没有 `block_table` 这个关键字参数。

**定位**：查 flash_attn 版本 = 2.4.x。

**根因**：`block_table` 参数（PagedAttention 寻址）是 flash_attn 较新版本才加的，2.4.x 没有。

**修复**：`pip install -U flash-attn --no-build-isolation` 升到 2.8.3（sm_120 需源码编译，15–30 分钟）。

**验证**：2.8.3 下 `block_table` 正常，对拍通过。

### Bug 3：模型权重下载连环坑

**现象**：`huggingface-cli` 命令提示已废弃；换 `hf` 命令后 hf-mirror.com 镜像 DNS 解析失败。

**定位**：huggingface-cli 被官方废弃改名 hf；国内镜像 hf-mirror 不稳定。

**修复**：改用 ModelScope（阿里）下载 Qwen3-0.6B。

**验证**：权重下载完整，`example.py` 能加载跑通。

### Bug 4：test 里 seqlens_k 形状对不上

**报错**：flash_attn 对拍跑挂，`seqlens_k must have shape (batch_size)`。

**定位**：第一个 flash_attn 对拍用例写 `num_seqs=4`，但 seq 长度列表只给了 3 个元素（漏了 `2048`）。

**根因**：`num_seqs` 声明和实际 seq 长度列表不一致，构造出来的 `seqlens_k` 张量 shape 不对。

**修复**：长度列表补全为 `[128, 257, 1000, 2048]`。

**验证**：对拍 2/2 通过。

### 环境约束：本地无 CUDA

**现象**：本地 macOS（arm64）跑不了 Triton/CUDA kernel——Triton 需要 NVIDIA GPU + Linux 驱动。

**影响**：所有 kernel 测试/benchmark 都必须在远程 Linux GPU 容器（RTX 5090 / 4090）上跑，本地只能写代码 + 传文件。

**应对**：本地写好 → 传远程容器 → 远程跑测试，结果贴回本地记文档。

---

## 十二、剩余工作

- **M4（可选加分）**：同样替换 prefill 的 `flash_attn_varlen_func`，彻底去掉 flash_attn 依赖。寒假做。
- **远程 GPU 已可退租**：代码在 GitHub + 本地，结果都已记录，退租不丢东西。
