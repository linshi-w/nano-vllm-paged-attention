# nano-vllm-triton-attention

Hand-written **Triton attention kernels** swapped into [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — a lightweight, vLLM-style LLM inference engine — removing the `flash_attn` dependency from both the **decode** and **prefill** stages.

## What this is

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (by [GeeeekExplorer](https://github.com/GeeeekExplorer)) is a clean, ~1,200-line vLLM implementation. Its attention layer originally delegates both prefill and decode to the `flash_attn` library.

This repo keeps the engine as-is and replaces **both attention paths** with hand-written Triton kernels:

- **Decode** → `paged_attention.py`: PagedAttention (block-based KV-cache addressing) + online softmax.
- **Prefill** → `prefill_attention.py`: FlashAttention-2-style kernel (varlen + causal + GQA + tensor core).

`flash_attn` now remains only on the prefix-cache prefill path (a paged-prefill kernel is the remaining step).

## Key contributions

### `nanovllm/layers/paged_attention.py` — Triton PagedAttention (decode)

- `grid = (num_seqs, num_heads)` — one program computes attention for one `(sequence, query head)` pair.
- **Paged address translation** — `block_table[seq][pos // kv_block_size]` maps a logical position to a physical KV block. A `BLOCK_SIZE = 64` tile never crosses a `kv_block_size = 256` boundary, so each tile needs only one table lookup.
- **GQA support** — `kv_head = head_idx // num_queries_per_kv`.
- **Online softmax** — maintains `(m_i, l_i, acc)` running states; no `[seq_len, seq_len]` score matrix, mathematically equivalent to a one-pass softmax.
- `bf16` in / `fp32` accumulate for numerical stability.

### `nanovllm/layers/prefill_attention.py` — Triton FlashAttention (prefill)

- `grid = (num_seq_blocks, num_heads)` — per-sequence query blocks; each program locates its `(seq_id, block_in_seq)` via a `cum_blocks` prefix sum and loops only within its own sequence.
- **Varlen + causal** — `cu_seqlens` packs variable-length sequences; inside a sequence the causal mask collapses to a simple `pos_n <= pos_m`.
- **GQA support** — `kv_head = head_idx // num_queries_per_kv`.
- **Tensor core** — `tl.dot` for both `QK^T` and `P·V` (prefill's `M = BLOCK_M >= 16`, unlike decode's single-token `M = 1`).
- **Online softmax** with a `-inf - -inf = NaN` guard for key blocks that are fully causally masked across sequences.

The engine change is confined to the prefill branch in `nanovllm/layers/attention.py`:

```python
# normal prefill
o = prefill_attention(q, k, v, context.cu_seqlens_q, self.scale)
```

## Results

**Correctness** — on RTX 4090 (Ada, sm_89): **7/7 decode** + **6/6 prefill**, each verified against a pure-PyTorch reference and `flash_attn`.

**Decode performance** — `triton.testing.do_bench`, 32 seqs × seq_len 4096, GQA 2:1 (head_dim 128):

| GPU | triton paged_attention | flash_attn_with_kvcache | gap |
|---|---|---|---|
| RTX 4090 (Ada, sm_89) | 633.4 us | 629.9 us | **parity (<1%)** |
| RTX 5090 (Blackwell, sm_120) | 404.3 us | 359.1 us | 12.6% slower |

**Prefill performance** — same shape, `flash_attn_varlen_func` as baseline:

| kernel | time | gap |
|---|---|---|
| triton prefill_attention | 17.3 ms | **122.6%** |
| flash_attn_varlen_func | 14.1 ms | — |

Decode is memory-bound (single-token queries read the full KV cache), so both kernels saturate bandwidth and the hand-written one reaches parity. Prefill is compute-bound, where `flash_attn`'s deep optimizations (diagonal block split, warp scheduling, smem management) keep the naive Triton kernel ~22% behind.

**End-to-end** (Qwen3-0.6B): **Prefill 8 tok/s, Decode 36 tok/s** with correct generations.

## Install & run

```bash
pip install torch triton transformers xxhash tqdm flash-attn

# download Qwen3-0.6B weights (example.py uses ~/huggingface/Qwen3-0.6B)
python example.py
```

## Tests

```bash
# correctness only (no flash_attn needed for the ref tests)
python test_paged_attention.py          # decode
python test_prefill_attention.py        # prefill

# + benchmark against flash_attn
python test_paged_attention.py --benchmark
python test_prefill_attention.py --benchmark
```

## Project layout

```
nanovllm/layers/paged_attention.py   # Triton PagedAttention (decode)
nanovllm/layers/prefill_attention.py # Triton FlashAttention (prefill)
nanovllm/layers/attention.py         # both paths now call the Triton kernels
test_paged_attention.py              # decode correctness + benchmark
test_prefill_attention.py            # prefill correctness + benchmark
```

## Acknowledgements

Based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by [GeeeekExplorer](https://github.com/GeeeekExplorer). See `LICENSE` for license terms.
