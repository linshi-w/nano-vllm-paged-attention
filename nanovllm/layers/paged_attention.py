"""
Triton PagedAttention kernel (decode 阶段).

替换 nano-vllm 里 decode 分支的 flash_attn_with_kvcache。

核心思路（PagedAttention 的地址翻译 + online softmax）：

1. 每个 program 处理一个 (seq, query_head) 对。grid = (num_seqs, num_heads)。
2. Q 只有一个 token，直接加载一整行 (head_dim,)。
3. KV cache 是分页存储的：逻辑位置 pos 对应的物理地址是
       block_table[seq][pos // kv_block_size] 块里的第 (pos % kv_block_size) 个 token。
   所以遍历 KV 时不是连续读，而是逐 tile 通过 block_table 查表找到物理块再读。
4. 每读一个 tile，用 online softmax 更新 (M, L, acc) 三个 running 状态，
   全程不需要 [seq_len, seq_len] 的 score 矩阵。

关于 decode 的 causal mask：decode 的 query 永远是序列里最新的那个 token，
它 attend 到所有 pos < seq_len 的位置，天然满足 causal，所以不需要额外 mask。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    Q, Out,
    K_cache, V_cache,
    BlockTable, SeqLens,
    softmax_scale,
    stride_q_seq, stride_q_head,
    stride_bt_seq, stride_bt_block,
    kv_block_size,          # 物理 block 大小（nano-vllm 里是 256）
    num_kv_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,       # kernel 内部 tile 大小，必须整除 kv_block_size
    NUM_QUERIES_PER_KV: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head = head_idx // NUM_QUERIES_PER_KV

    offs_d = tl.arange(0, HEAD_DIM)
    # 加载这一行 query：(head_dim,)，转 fp32 累加
    q = tl.load(Q + seq_idx * stride_q_seq + head_idx * stride_q_head + offs_d).to(tl.float32)

    seq_len = tl.load(SeqLens + seq_idx)
    num_tiles = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # KV cache 布局：(num_blocks, kv_block_size, num_kv_heads, head_dim)
    stride_block = kv_block_size * num_kv_heads * HEAD_DIM
    stride_token = num_kv_heads * HEAD_DIM
    stride_head = HEAD_DIM

    offs_j = tl.arange(0, BLOCK_SIZE)

    # online softmax 三个 running 状态
    m_i = tl.full([], float("-inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros([], dtype=tl.float32)                  # exp 累加和
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)          # 加权 V 累加

    for tile_idx in tl.range(0, num_tiles):
        start = tile_idx * BLOCK_SIZE
        pos = start + offs_j                       # 全局 token 位置
        valid = pos < seq_len                      # 处理最后一个不完整的 block

        # PagedAttention 地址翻译：逻辑位置 -> 物理 (block, offset)
        block_idx = start // kv_block_size
        offset_in_block = start % kv_block_size
        block_id = tl.load(BlockTable + seq_idx * stride_bt_seq + block_idx * stride_bt_block)

        kv_base = block_id * stride_block + offset_in_block * stride_token + kv_head * stride_head
        k_ptrs = K_cache + kv_base + offs_j[:, None] * stride_token + offs_d[None, :]
        v_ptrs = V_cache + kv_base + offs_j[:, None] * stride_token + offs_d[None, :]

        K = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)   # (BLOCK_SIZE, HEAD_DIM)
        s = tl.sum(q[None, :] * K, axis=1) * softmax_scale                   # (BLOCK_SIZE,)
        s = tl.where(valid, s, float("-inf"))

        # online softmax 更新
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)                 # 修正因子
        p = tl.exp(s - m_new)                       # (BLOCK_SIZE,)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        V = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * V, axis=0)   # (HEAD_DIM,)
        m_i = m_new

    o = acc / l_i
    tl.store(Out + seq_idx * stride_q_seq + head_idx * stride_q_head + offs_d, o.to(Out.dtype.element_ty))


def paged_attention(
    q: torch.Tensor,              # (num_seqs, num_heads, head_dim)
    k_cache: torch.Tensor,        # (num_blocks, kv_block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,
    block_table: torch.Tensor,    # (num_seqs, max_blocks) int32, -1 填充
    seq_lens: torch.Tensor,       # (num_seqs,) int32
    softmax_scale: float,
) -> torch.Tensor:                # (num_seqs, num_heads, head_dim)
    num_seqs, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    kv_block_size = k_cache.shape[1]
    assert num_heads % num_kv_heads == 0
    num_q_per_kv = num_heads // num_kv_heads
    assert head_dim & (head_dim - 1) == 0, "head_dim 必须是 2 的幂"

    BLOCK_SIZE = 64
    assert kv_block_size % BLOCK_SIZE == 0, "kv_block_size 必须是 BLOCK_SIZE 的整数倍"

    q = q.contiguous()
    out = torch.empty_like(q)
    grid = (num_seqs, num_heads)
    _paged_attention_kernel[grid](
        q, out, k_cache, v_cache, block_table, seq_lens,
        softmax_scale,
        q.stride(0), q.stride(1),
        block_table.stride(0), block_table.stride(1),
        kv_block_size,
        num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_QUERIES_PER_KV=num_q_per_kv,
        num_warps=4,
    )
    return out
