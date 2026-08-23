"""
Triton FlashAttention-2 风格 prefill kernel（varlen + causal + tensor core）。

替换 nano-vllm 里 prefill 分支的 flash_attn_varlen_func（普通 prefill，无 prefix cache）。

和 decode 的 paged_attention 关键区别：
    decode 的 Q 只有一个 token（M=1），tl.dot 被 >=16 约束卡死，用不了 tensor core；
    prefill 的 Q 是一整批 token（M = BLOCK_M >= 16），QK^T 和 P·V 都能走 tensor core。

按序列分块：grid 第 0 维是「序列内的 query block」，每个 program 明确属于一个序列，
循环只在该序列内迭代 key block，causal 退化成纯 `pos_n <= pos_m`，
避免循环内反复做 seq_id 的广播比较 + cu_seqlens gather。

地址/形状约定（普通 prefill，cu_seqlens_q == cu_seqlens_k，无 paging）：
    Q: (total_tokens, num_heads, head_dim)
    K/V: (total_tokens, num_kv_heads, head_dim)
    连续 varlen 打包。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_attention_kernel(
    Q, K, V, Out,
    cu_seqlens,          # (B+1,) int32
    cum_blocks,          # (B,) int32 — 每个 seq 的 query block 计数前缀和
    softmax_scale,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_vt, stride_vh,
    stride_ot, stride_oh,
    B,                   # num_seqs (runtime)
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

    # seq_id = count(cum_blocks[i] <= pid_m)，即 pid_m 落在哪个序列的 block 区间
    cb = tl.load(cum_blocks + tl.arange(0, MAX_SEQS),
                 mask=tl.arange(0, MAX_SEQS) < B, other=1 << 30)
    seq_id = tl.sum((pid_m >= cb).to(tl.int32))
    prev_blocks = tl.load(cum_blocks + seq_id - 1, mask=seq_id > 0, other=0)
    block_in_seq = pid_m - prev_blocks

    seq_start = tl.load(cu_seqlens + seq_id)
    seq_end = tl.load(cu_seqlens + seq_id + 1)

    start_m = seq_start + block_in_seq * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    pos_m = offs_m - seq_start

    q_valid = offs_m < seq_end
    q_ptrs = Q + offs_m[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # 序列内 causal：本 block 只需 attend 到 <= 本 block 最大位置的 key
    hi = tl.minimum(start_m + BLOCK_M, seq_end)
    num_n_blocks = tl.cdiv(hi - seq_start, BLOCK_N)

    for bn in tl.range(0, num_n_blocks):
        offs_n = seq_start + bn * BLOCK_N + tl.arange(0, BLOCK_N)
        pos_n = offs_n - seq_start

        k_ptrs = K + offs_n[:, None] * stride_kt + kv_h * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_end, other=0.0)

        causal = pos_n[None, :] <= pos_m[:, None]
        causal = causal & (offs_n[None, :] < seq_end)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(causal, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.where(m_new > float("-inf"), tl.exp(m_i - m_new), 0.0)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(m_new[:, None] > float("-inf"), p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + offs_n[:, None] * stride_vt + kv_h * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_end, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = Out + offs_m[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_valid[:, None])


def prefill_attention(
    q,                    # (total_tokens, num_heads, head_dim)
    k,                    # (total_tokens, num_kv_heads, head_dim)
    v,
    cu_seqlens,           # (B+1,) int32
    softmax_scale,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:        # (total_tokens, num_heads, head_dim)
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    B = cu_seqlens.shape[0] - 1
    assert num_heads % num_kv_heads == 0
    num_q_per_kv = num_heads // num_kv_heads

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    blocks_per_seq = (seq_lens + BLOCK_M - 1) // BLOCK_M
    cum_blocks = torch.cumsum(blocks_per_seq, 0).to(torch.int32)
    total_blocks = int(cum_blocks[-1].item())

    MAX_SEQS = triton.next_power_of_2(B)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    grid = (total_blocks, num_heads)
    _prefill_attention_kernel[grid](
        q, k, v, out, cu_seqlens, cum_blocks, softmax_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        B,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        MAX_SEQS=MAX_SEQS,
        NUM_QUERIES_PER_KV=num_q_per_kv,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
