"""
Triton paged prefill kernel（prefix cache 场景）。

替换 nano-vllm 里 prefix cache prefill 分支的 flash_attn_varlen_func（带 block_table）。

和普通 prefill（prefill_attention.py）的区别：
    - K/V 不是连续 varlen 打包，而是分页存在 KV cache 里，靠 block_table 寻址。
    - K 的逻辑长度 seqlen_k >= Q 的 seqlen_q，多出来的 cached 是命中缓存的前缀。
    - Q 的第 j 个 token 在 K 逻辑序列里的绝对位置 = cached + j（不是 j），
      所以 causal mask 用绝对位置比较。

按序列分块（同 prefill_attention）：grid 第 0 维是「序列内的 query block」，
K 迭代在序列内相对位置 [0, seqlen_k) 上进行。BLOCK_N 整除 kv_block_size
保证每个 K tile 不跨物理块，tile 只需查一次 block_table（同 decode 的 paged_attention）。

地址/形状约定（prefix cache prefill）：
    Q: (total_q_tokens, num_heads, head_dim)        连续打包，只含未缓存的新 token
    K/V: (num_blocks, kv_block_size, num_kv_heads, head_dim)  分页 KV cache
    cu_seqlens_q: (B+1,) Q 累积长度；cu_seqlens_k: (B+1,) K 累积长度（含 cached）
    block_table: (B, max_blocks) int32，-1 填充
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_prefill_attention_kernel(
    Q, K_cache, V_cache, Out,
    cu_seqlens_q, cu_seqlens_k,
    cum_blocks_q,          # (B,) int32 — 每个 seq 的 query block 计数前缀和
    block_table,           # (B, max_blocks) int32
    softmax_scale,
    stride_qt, stride_qh,
    stride_ot, stride_oh,
    stride_bt_seq, stride_bt_block,
    kv_block_size,         # 物理 block 大小（nano-vllm 里是 256）
    num_kv_heads,
    B,                     # num_seqs (runtime)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    NUM_QUERIES_PER_KV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // NUM_QUERIES_PER_KV

    offs_d = tl.arange(0, HEAD_DIM)

    # seq_id = count(cum_blocks_q[i] <= pid_m)，即 pid_m 落在哪个序列的 block 区间
    cb = tl.load(cum_blocks_q + tl.arange(0, MAX_SEQS),
                 mask=tl.arange(0, MAX_SEQS) < B, other=1 << 30)
    seq_id = tl.sum((pid_m >= cb).to(tl.int32))
    prev_blocks = tl.load(cum_blocks_q + seq_id - 1, mask=seq_id > 0, other=0)
    block_in_seq = pid_m - prev_blocks

    q_start = tl.load(cu_seqlens_q + seq_id)
    q_end = tl.load(cu_seqlens_q + seq_id + 1)
    k_start = tl.load(cu_seqlens_k + seq_id)
    k_end = tl.load(cu_seqlens_k + seq_id + 1)
    seqlen_q = q_end - q_start
    seqlen_k = k_end - k_start
    cached = seqlen_k - seqlen_q                    # 命中缓存的前缀 token 数

    start_m = q_start + block_in_seq * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    q_valid = offs_m < q_end
    pos_q = cached + (offs_m - q_start)             # 绝对位置（K 逻辑序列内）

    q_ptrs = Q + offs_m[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # KV cache 布局 (num_blocks, kv_block_size, num_kv_heads, head_dim)
    stride_block = kv_block_size * num_kv_heads * HEAD_DIM
    stride_token = num_kv_heads * HEAD_DIM
    stride_head = HEAD_DIM

    offs_j = tl.arange(0, BLOCK_N)

    num_n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    for bn in tl.range(0, num_n_blocks):
        n_start = bn * BLOCK_N                      # K 序列内相对位置
        pos_n = n_start + offs_j
        k_valid = pos_n < seqlen_k

        # 地址翻译：逻辑位置 -> 物理 (block, offset)
        block_idx = n_start // kv_block_size
        offset_in_block = n_start % kv_block_size
        block_id = tl.load(block_table + seq_id * stride_bt_seq + block_idx * stride_bt_block)
        kv_base = block_id * stride_block + offset_in_block * stride_token + kv_h * stride_head

        k_ptrs = K_cache + kv_base + offs_j[:, None] * stride_token + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)

        causal = pos_n[None, :] <= pos_q[:, None]
        causal = causal & k_valid[None, :]

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(causal, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.where(m_new > float("-inf"), tl.exp(m_i - m_new), 0.0)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(m_new[:, None] > float("-inf"), p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V_cache + kv_base + offs_j[:, None] * stride_token + offs_d[None, :]
        v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = Out + offs_m[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_valid[:, None])


def paged_prefill_attention(
    q,                    # (total_q_tokens, num_heads, head_dim)
    k_cache,              # (num_blocks, kv_block_size, num_kv_heads, head_dim)
    v_cache,
    cu_seqlens_q,         # (B+1,) int32
    cu_seqlens_k,         # (B+1,) int32
    block_table,          # (B, max_blocks) int32, -1 填充
    softmax_scale,
    BLOCK_M: int = 128,
    BLOCK_N: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:        # (total_q_tokens, num_heads, head_dim)
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_block_size = k_cache.shape[1]
    B = cu_seqlens_q.shape[0] - 1
    assert num_heads % num_kv_heads == 0
    num_q_per_kv = num_heads // num_kv_heads
    assert kv_block_size % BLOCK_N == 0, "BLOCK_N 必须整除 kv_block_size，否则 K tile 跨物理块"

    seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    blocks_per_seq = (seq_lens_q + BLOCK_M - 1) // BLOCK_M
    cum_blocks_q = torch.cumsum(blocks_per_seq, 0).to(torch.int32)
    total_blocks = int(cum_blocks_q[-1].item())

    MAX_SEQS = triton.next_power_of_2(B)

    q = q.contiguous()
    out = torch.empty_like(q)
    grid = (total_blocks, num_heads)
    _paged_prefill_attention_kernel[grid](
        q, k_cache, v_cache, out,
        cu_seqlens_q, cu_seqlens_k, cum_blocks_q, block_table,
        softmax_scale,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        block_table.stride(0), block_table.stride(1),
        kv_block_size, num_kv_heads,
        B,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        MAX_SEQS=MAX_SEQS,
        NUM_QUERIES_PER_KV=num_q_per_kv,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
