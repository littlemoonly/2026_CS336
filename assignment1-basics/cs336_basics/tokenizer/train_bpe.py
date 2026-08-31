from __future__ import annotations
import os
import regex as re
import multiprocessing
from tokenizer.utils import find_chunk_boundaries
from dataclasses import dataclass

# uv run pytest tests/test_train_bpe.py
DEBUG = False

@dataclass
class Node:
    ''' 表示 seq 中的每一个 token bytes'''
    token_id: int
    prev: 'Node | None' = None  # 可选字段（有默认值 None）
    next: 'Node | None' = None


def worker(start, end, input_path, special_tokens, pattern, byte_to_id):
    '''
    返回局部的 {token_tuple: count} 字典
    note: 对 input_path 和 byte_to_id 都是只读，因此线程安全
    '''
    chunk_seq = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        raw_bytes = f.read(end - start)
        text = raw_bytes.decode("utf-8", errors="ignore")

        # 按 special tokens 切分文本（防止跨文档边界合并）
        if len(special_tokens) > 0:
            delim = "|".join(re.escape(tok) for tok in special_tokens)
            docs = re.split(delim, text)
        else:
            docs = [text]

        for doc in docs:
            if not doc:
                continue
            # 对每个文档做预分词，把每个 pre-token 编码为 token ID 序列
            for match in re.finditer(pattern, doc):
                pre_token_str = match.group()      # 默认是 group(0), 整个 pre-token 字符串
                pre_token_bytes = pre_token_str.encode("utf-8")
                token_ids = tuple(byte_to_id[bytes([b])] for b in pre_token_bytes)
                chunk_seq[token_ids] = chunk_seq.get(token_ids, 0) + 1
    return chunk_seq


def get_pretok_freq(input_path, special_tokens, pattern, byte_to_id, num_processes=4):
    '''返回 seqs {tuple_of_token_ids: count}，统计每个不同的 token ID 序列在语料中出现了多少次 '''
    seqs = {}

    with open(input_path, "rb") as f:
        # 先按 <|endoftext|> 边界做 chunk 划分，默认用 <|endoftext|> 划分 chunk
        boundaries = find_chunk_boundaries(f, desired_num_chunks=4, split_special_token=b"<|endoftext|>")

        # 构建参数列表：每个元素是一个 tuple，包含这个 chunk 需要的所有信息
        arg_list = [(start, end, input_path, special_tokens, pattern, byte_to_id) for  start, end in zip(boundaries[:-1], boundaries[1:])]

        with multiprocessing.Pool(processes=num_processes) as pool:
            all_chunk_seqs = pool.starmap(worker, arg_list)

    # 合并处理所有 key token tuple
    for chunk_seq in all_chunk_seqs:
        for key, count in chunk_seq.items():
            seqs[key] = seqs.get(key, 0) + count

    return seqs


def get_seq_head_node(token_ids:tuple):
    head = None
    prev = None
    for token_id in token_ids:
        node = Node(token_id=token_id)
        node.prev = prev
        if prev is not None:
            prev.next = node
        else:
            head = node
        prev = node
    return head


def merge_node(lnode:Node, new_token_id):
    ''' 将 lnode 和 其后的node 合并, 原地把左边的node变成新的node '''
    assert lnode.next is not None, "merge_node: lnode.next is None!!!"
    lnode.next.token_id = -1
    lnode.token_id = new_token_id
    lnode.next = lnode.next.next
    if lnode.next is not None:
        lnode.next.prev = lnode


def do_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab = {}
    merges = []

    # 初始化 vocab: {token_id(int): bytes}, 包含: 256 个单字节 (ID 0~255) + 所有 special_tokens
    next_id = 0     # next_id = len(vocab)
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    byte_to_id = {v: k for k, v in vocab.items()}   # 反向映射: bytes → id

    # 总共需要多少次合并
    merge_times = vocab_size - len(vocab)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    seqs_tup = get_pretok_freq(input_path, special_tokens, PAT, byte_to_id) # {tuple(tokenids), count}

    seqs_node = {}  # seq_id -> (head_node, count)
    pair_counts = {}    # {(token_id_A, token_id_B): total_count}
    seq_id = 0
    # 记录每一个 token pair 的位置索引：在哪个 seq 的哪个起始位置
    # {key: (tokenid_a, tokenid_b) -> value: list[tuple(seq_id, Node)]}
    pair_cache = {}

    # 同步维护 seq_node, pair_count, pair_cache
    for token_ids, count in seqs_tup.items():

        head = get_seq_head_node(token_ids) # 创建该 seq 的链表，返回头节点
        seqs_node[seq_id] = (head, count)
        
        cur = head
        while cur.next is not None: # cur 即 tokenid_A 对应的节点引用
            token_pair = (cur.token_id, cur.next.token_id)
            pair_counts[token_pair] = pair_counts.get(token_pair, 0) + count
            pair_cache.setdefault(token_pair, []).append((seq_id, cur))
            cur = cur.next

        seq_id += 1
    

    for merge_time in range(merge_times):
        if len(pair_counts) == 0:
            # （小）语料中的 pair 被合并完，用作调试
            break

        best_pair = max(pair_counts, key=lambda p: (    # 这里 p 是 tuple(token_id1, token_id2)
            pair_counts[p],
            vocab[p[0]], # 字典序越大越好
            vocab[p[1]],      
        ))


        token_id_A, token_id_B = best_pair[0], best_pair[1]
        vocab[next_id] = vocab[token_id_A] + vocab[token_id_B]

        # note: 两个 bytes token 拼接后可能不是完整的 utf-8 序列，因此不能用 decode 调试输出
        if DEBUG is True:
            print(f"merge_time: {merge_time}")
            print(f"[DEBUG] best pair: {best_pair}, {repr(vocab[next_id])}, cnt= {pair_counts[best_pair]}")

        new_token_id = next_id
        next_id += 1
        merges.append((vocab[token_id_A], vocab[token_id_B]))

        # 取出每一个含有 best pair 的 seq
        for seq_id, pos_node in pair_cache[best_pair]:
            head, seq_count = seqs_node[seq_id][0], seqs_node[seq_id][1] # 这个 seq 的头节点、个数

            if pos_node.token_id < 0:
                continue

            old_right = pos_node.next
            merge_node(pos_node, new_token_id)

            # 更新 seq_node 的头节点
            # if pos_node.prev is None:   # 其实 A node 引用没改，不用这个分支
            #     seqs_node[seq_id] = pos_node

            # (lnode, [pos_node, pos_node.next], rnode), 已合并，posnode 为新的node
            if pos_node.prev is not None:
                # 维护 pair_count
                # （1）删 (l, a)
                left_id = pos_node.prev.token_id
                pair_counts[(left_id, token_id_A)] -= seq_count
                if pair_counts[(left_id, token_id_A)] <= 0:
                    del pair_counts[(left_id, token_id_A)]

                #  (2) 加 (l, new)
                new_lpair = (left_id, new_token_id)
                pair_counts[new_lpair] = pair_counts.get(new_lpair, 0) + seq_count

                # 维护 pair_cache
                # （1）删 (l, a)
                pair_cache[(left_id, token_id_A)].remove((seq_id, pos_node.prev))
                if len(pair_cache[(left_id, token_id_A)]) == 0:
                    del pair_cache[(left_id, token_id_A)]
                #  (2) 加 (l, new)
                pair_cache.setdefault(new_lpair, []).append((seq_id, pos_node.prev))

            if pos_node.next is not None:
                right_id = pos_node.next.token_id
                pair_counts[(token_id_B, right_id)] -= seq_count
                if pair_counts[(token_id_B, right_id)] <= 0:
                    del pair_counts[(token_id_B, right_id)]

                new_rpair = (new_token_id, right_id)
                pair_counts[new_rpair] = pair_counts.get(new_rpair, 0) + seq_count

                pair_cache[(token_id_B, right_id)].remove((seq_id, old_right))
                if len(pair_cache[(token_id_B, right_id)]) == 0:
                    del pair_cache[(token_id_B, right_id)]

                pair_cache.setdefault(new_rpair, []).append((seq_id, pos_node))
                
        del pair_cache[best_pair]
        del pair_counts[best_pair]

        if DEBUG:
            print(f"[DEBUG] pair_counts:{pair_counts}")

    return vocab, merges