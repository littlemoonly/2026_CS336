'''
Serialize the resulting vocabulary and merges to disk for further inspection
'''

from tests.adapters import run_train_bpe
from tests.common import gpt2_bytes_to_unicode
import json

INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
# INPUT_PATH = "tests/fixtures/corpus.en"
VOCAB_OUTPUT_PATH = "cs336_basics/tinystories_bpe_vocab.json"
MERGES_OUTPUT_PATH = "cs336_basics/tinystories_bpe_merges.txt"

def bytes2str(token_bytes: bytes, gpt2_int2unicode:dict):
    token_str = ""
    # print(f"[DEBUG] token_bytes = {token_bytes}")
    for b in token_bytes:
        # print(f"    [DEBUG] {b}  {gpt2_int2unicode[b]}")
        token_str += gpt2_int2unicode[b]
    return token_str

def train_bpe_tinystories():
    vocab, merges = run_train_bpe(
        input_path=INPUT_PATH,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # gpt2_byte_decoder = {bytes([k]) : v for k, v in gpt2_bytes_to_unicode().items()}
    gpt2_int2unicode = gpt2_bytes_to_unicode() # dict{int -> str}

    vocab_data = {}
    for token_id, token_bytes in vocab.items():
        # vocab 字典：{tokenid(int) -> bytes对象}
        token_str = bytes2str(token_bytes, gpt2_int2unicode)
        vocab_data[token_str] = token_id
        # print(f"[DEBUG] add vocab_data item: {token_str} -> {token_id}")

    with open(VOCAB_OUTPUT_PATH, "w") as f:
        # 输出：{"A": 33, ..., "Ġt": 257,}
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    with open(MERGES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        # merges: list[(id1, id2)]
        for id1, id2 in merges:
            f.writelines(f"{bytes2str(id1, gpt2_int2unicode)} {bytes2str(id2, gpt2_int2unicode)}\n")

if __name__ == '__main__':
    train_bpe_tinystories()