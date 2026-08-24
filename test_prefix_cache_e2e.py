"""
端到端验证 paged prefill（prefix cache）真正被触发且生成正常。

做法：构造两个共享 >=256 token 前缀的 prompt，分两次 generate。
第一次 prompt1 跑完注册缓存；第二次 prompt2 命中 prompt1 的前缀 block，
prefill 时 cu_seqlens_k > cu_seqlens_q，block_tables 非 None，走 paged prefill 路径。

通过 monkeypatch 计数确认 paged_prefill_attention 被调用。
"""

import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# monkeypatch 计数：确认 paged prefill 被调用
from nanovllm.layers import attention as attn_mod

_calls = {"n": 0}
_orig = attn_mod.paged_prefill_attention


def _counted(*args, **kwargs):
    _calls["n"] += 1
    return _orig(*args, **kwargs)


attn_mod.paged_prefill_attention = _counted


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sp = SamplingParams(temperature=0.6, max_tokens=24)

    base = "The quick brown fox jumps over the lazy dog. "
    prefix = base * 40                                    # 约 360 tokens，> 256（一个 block）
    prompt1 = prefix + "Now tell me a short story."
    prompt2 = prefix + "Now write a short poem."

    n_prefix_tokens = len(tokenizer.encode(prefix))
    print(f"共享前缀 token 数: {n_prefix_tokens}（需 >= 256）")

    out1 = llm.generate([prompt1], sp)
    print(f"prompt1 完成，paged_prefill 调用次数 = {_calls['n']}")
    print("prompt1 输出:", repr(out1[0]["text"][-60:]))

    out2 = llm.generate([prompt2], sp)
    print(f"prompt2 完成，paged_prefill 调用次数 = {_calls['n']}")
    print("prompt2 输出:", repr(out2[0]["text"][-60:]))

    assert _calls["n"] >= 1, "paged prefill 没有被触发！"
    print(f"\nOK: paged prefill 被调用了 {_calls['n']} 次，端到端 prefix cache 走通。")


if __name__ == "__main__":
    main()
