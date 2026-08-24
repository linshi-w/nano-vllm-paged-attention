# M2 验证结果 —— 接入 engine 端到端跑通

- 日期：2026-08-15
- 环境：RTX 5090 (sm_120)，flash_attn 2.8.3.post1，torch 2.6.0a0 (cu128)
- 结果：✅ `example.py` 端到端跑通，生成两段文本
- 初步性能：Prefill=7tok/s, Decode=61tok/s（未调优）

## 验证内容

decode 阶段改用自研 Triton PagedAttention kernel（`attention.py` 第 73 行 `o = paged_attention(...)`），prefill 仍用 `flash_attn_varlen_func`。整个引擎（scheduler → model_runner → 前向 → 采样）端到端正常生成文本，无报错。

## 输出示例

- Prompt 1 "introduce yourself" → 生成连贯英文自我介绍
- Prompt 2 "list all prime numbers within 100" → 生成回答（带 `<think>` 推理块，因 max_tokens 上限截断）

## 备注

- 第二段被截断 = Qwen3-0.6B 是 thinking 模型，`<think>` 推理块消耗 token 预算 + max_tokens 上限，与 kernel 无关。
- 下一步 M3：benchmark 对比 flash_attn_with_kvcache（寒假做）。
