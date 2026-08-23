# nano-vllm-paged-attention

A from-scratch **Triton PagedAttention kernel** swapped into [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — a lightweight, vLLM-style LLM inference engine — removing the `flash_attn` dependency from the decode stage.

## What this is

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (by [GeeeekExplorer](https://github.com/GeeeekExplorer)) is a clean, ~1,200-line vLLM implementation. Its attention layer originally delegates both prefill and decode to the `flash_attn` library.

This repo keeps the engine as-is and replaces **only the decode path** with a hand-written Triton kernel that implements **PagedAttention** — the block-based KV-cache addressing vLLM uses in production — plus online-softmax, entirely in Triton.

## Key contribution

**`nanovllm/layers/paged_attention.py`** — a Triton PagedAttention kernel for the decode stage:

- `grid = (num_seqs, num_heads)` — one program computes attention for one `(sequence, query head)` pair.
- **Paged address translation** — `block_table[seq][pos // kv_block_size]` maps a logical position to a physical KV block. A `BLOCK_SIZE = 64` tile never crosses a `kv_block_size = 256` boundary, so each tile needs only one table lookup.
- **GQA support** — `kv_head = head_idx // num_queries_per_kv`.
- **Online softmax** — maintains `(m_i, l_i, acc)` running states; no `[seq_len, seq_len]` score matrix, mathematically equivalent to a one-pass softmax.
- `bf16` in / `fp32` accumulate for numerical stability.

The only change to the engine is a single call-site in `nanovllm/layers/attention.py`:

```python
# decode
o = paged_attention(q, k_cache, v_cache,
                    context.block_tables, context.context_lens,
                    self.scale)
```

## Results

**Correctness** — verified on an RTX 5090 (Blackwell, sm_120):
- **5/5** against a pure-PyTorch reference (MHA full/partial blocks, GQA 2:1, GQA 4:1, small block size)
- **2/2** against `flash_attn_with_kvcache` (GQA 2:1, MHA)

**Performance** — `triton.testing.do_bench`, 32 seqs × seq_len 4096, GQA 2:1 (head_dim 128):

| GPU | triton paged_attention | flash_attn_with_kvcache | gap |
|---|---|---|---|
| RTX 4090 (Ada, sm_89) | 633.4 us | 629.9 us | **parity (<1%)** |
| RTX 5090 (Blackwell, sm_120) | 404.3 us | 359.1 us | 12.6% slower |

On the 4090 the hand-written kernel **matches `flash_attn_with_kvcache`** (within measurement noise). On the 5090 it's ~12.6% slower — `flash_attn` ships Blackwell-specific optimizations the naive Triton kernel doesn't use. Decode attention is memory-bound (single-token queries read the full KV cache), so on Ada, where both kernels saturate bandwidth, the Triton kernel reaches parity.

End-to-end inference runs correctly with the custom kernel in the decode path: **Prefill 7 tok/s, Decode 61 tok/s** (unoptimized).

## Install & run

```bash
pip install torch triton transformers xxhash tqdm flash-attn

# download Qwen3-0.6B weights (example.py uses ~/huggingface/Qwen3-0.6B)
python example.py
```

## Tests

```bash
# correctness only (no flash_attn needed)
python test_paged_attention.py

# + benchmark against flash_attn_with_kvcache
python test_paged_attention.py --benchmark
```

## Project layout

```
nanovllm/layers/paged_attention.py   # Triton PagedAttention kernel (this repo's core)
nanovllm/layers/attention.py         # decode path now calls paged_attention()
test_paged_attention.py              # correctness tests + optional benchmark
```

## Acknowledgements

Based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by [GeeeekExplorer](https://github.com/GeeeekExplorer). See `LICENSE` for license terms.
