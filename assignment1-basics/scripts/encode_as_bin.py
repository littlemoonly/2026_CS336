# 把 token ID 数组保存为二进制文件
from cs336_basics.tokenizer.bpe import Tokenizer
from pathlib import Path
import numpy as np
from collections.abc import Iterator


def read_blocks_by_eot(
    file_path: str,
    read_size: int = 1024 * 1024,
    eot_token: str = "<|endoftext|>",
) -> Iterator[str]:
    """
    按 <|endoftext|> 分块读取文本。

    每次 yield 的文本都包含末尾的 <|endoftext|>。
    最后一段如果没有结束标记，也会被返回。
    """
    buffer = ""

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(read_size)

            if not chunk:
                break

            buffer += chunk

            while True:
                eot_index = buffer.find(eot_token)

                if eot_index == -1:
                    break

                block_end = eot_index + len(eot_token)

                # 包含 <|endoftext|>
                block = buffer[:block_end]
                yield block

                # 删除已经处理的部分
                buffer = buffer[block_end:]

    # 文件最后可能没有 <|endoftext|>
    if buffer:
        yield buffer


def encode_as_bin_by_eot(
    tokenizer: Tokenizer,
    data_path: str,
    dest_path: str,
    read_size: int = 1024 * 1024,
) -> int:
    total_tokens = 0

    # 整个任务开始时使用 wb，清空或创建目标文件
    with open(dest_path, "wb") as bin_file:
        for block in read_blocks_by_eot(
            file_path=data_path,
            read_size=read_size,
        ):
            token_ids = np.asarray(
                tokenizer.encode(block),
                dtype=np.int64,
            )

            # bin_file 已经打开，后续数组会依次写到文件末尾
            token_ids.tofile(bin_file)

            total_tokens += token_ids.size

    return total_tokens

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":

    tokenizer = Tokenizer.from_files(
        vocab_filepath=PROJECT_ROOT / "assets" / "tinystories_bpe_vocab.json",
        merges_filepath=PROJECT_ROOT / "assets" / "tinystories_bpe_merges.txt",
        special_tokens=['<|endoftext|>'],
    )

    train_token_count = encode_as_bin_by_eot(
        tokenizer=tokenizer,
        data_path=PROJECT_ROOT / "data" / "just_one_story.txt",
        dest_path=PROJECT_ROOT / "data" / "just_one_story.bin",
    )

    # valid_token_count = encode_as_bin_by_eot(
    #     tokenizer=tokenizer,
    #     data_path=PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
    #     dest_path=PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.bin",
    # )

    print(f"train tokens: {train_token_count:,}")
    # print(f"valid tokens: {valid_token_count:,}")