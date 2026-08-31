import json
from collections.abc import Iterable, Iterator
import sys
import os
import regex as re

from tests.common import gpt2_bytes_to_unicode

GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    """
    Byte-level BPE tokenizer.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)

        # merge 排名越小，优先级越高
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.byte_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # 去重并保持原顺序
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.special_to_id: dict[str, int] = {}

        # 如果特殊 token 不在 vocab 中，就追加进去
        next_id = max(self.vocab, default=-1) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            token_id = self.byte_to_id.get(token_bytes)

            if token_id is None:
                token_id = next_id
                next_id += 1

                self.vocab[token_id] = token_bytes
                self.byte_to_id[token_bytes] = token_id

            self.special_to_id[token] = token_id

        # 长 token 放前面，保证重叠时优先匹配最长者
        ordered_specials = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        self.special_pattern = (
            re.compile(
                "|".join(re.escape(token) for token in ordered_specials)
            )
            if ordered_specials
            else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=None,
    ):
        """根据 GPT-2 格式的 vocab 和 merges 创建 tokenizer。"""

        byte_decoder = {
            char: byte
            for byte, char in gpt2_bytes_to_unicode().items()
        }

        def decode_gpt2_token(token: str) -> bytes:
            return bytes(byte_decoder[char] for char in token)

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            str_to_id = json.load(f)

        vocab = {
            int(token_id): decode_gpt2_token(token)
            for token, token_id in str_to_id.items()
        }

        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#version:"):
                    continue

                parts = line.split()

                if len(parts) != 2:
                    continue

                left, right = parts
                merges.append(
                    (
                        decode_gpt2_token(left),
                        decode_gpt2_token(right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _split_special(
        self,
        text: str,
    ) -> Iterator[tuple[bool, str]]:
        """
        将文本切成：
        (是否为特殊 token, 文本片段)
        """

        if self.special_pattern is None:
            if text:
                yield False, text
            return

        start = 0

        for match in self.special_pattern.finditer(text):
            if match.start() > start:
                yield False, text[start:match.start()]

            yield True, match.group(0)
            start = match.end()

        if start < len(text):
            yield False, text[start:]

    def _bpe(self, piece: bytes) -> list[bytes]:
        """对一个 pre-token 执行 BPE 合并。"""

        tokens = [bytes([byte]) for byte in piece]

        while len(tokens) > 1:
            best_pair = None
            best_rank = None

            # 找当前相邻 token 中优先级最高的 pair
            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)

                if rank is not None and (
                    best_rank is None or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            # 一次合并该 pair 的所有非重叠位置
            merged_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    merged_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1

            tokens = merged_tokens

        return tokens

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        """编码不包含特殊 token 的普通文本。"""

        for match in GPT2_PATTERN.finditer(text):
            piece = match.group(0).encode("utf-8")

            for token_bytes in self._bpe(piece):
                yield self.byte_to_id[token_bytes]

    def _encode(self, text: str) -> Iterator[int]:
        """encode 和 encode_iterable 共用的惰性编码逻辑。"""

        for is_special, piece in self._split_special(text):
            if is_special:
                yield self.special_to_id[piece]
            else:
                yield from self._encode_ordinary(piece)

    def encode(self, text: str) -> list[int]:
        """将字符串编码为 token ID。"""

        return list(self._encode(text))

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        Lazily encode strings and yield token IDs.
        """
        for text in iterable:
            yield from self._encode(text)

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """逐块编码字符串迭代器，不加载全部文本。"""

        for text in iterable:
            yield from self._encode(text)

    def decode(self, ids: list[int]) -> str:
        """将 token ID 解码为字符串。"""

        data = b"".join(self.vocab[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")

if __name__=='__main__':
    print(sys.path)
    print(os.getcwd())
    VOCAB_INPUT_FILE = 'assets/tinystories_bpe_vocab.json'
    MERGE_INPUT_FILE = 'assets/tinystories_bpe_merges.txt'
    special_tokens = ['<|endoftext|>']
    tokenizer = Tokenizer.from_files(VOCAB_INPUT_FILE, MERGE_INPUT_FILE, special_tokens)

    print(tokenizer.vocab)
    ids = [10, 430, 439, 259, 398, 401, 283, 259, 390, 496, 402, 551, 46, 551, 502, 266, 1425, 263, 1569, 623, 474, 46, 316, 382, 661, 1783, 624, 44, 516, 995, 2687, 115, 381, 405, 354, 4083, 313, 259, 1107, 46, 527, 327, 44, 551, 283, 1280, 1472, 263, 1107, 629, 286, 550, 2269, 259, 378, 850, 2687, 46, 936, 551, 382, 309, 286, 283]

    print(tokenizer.decode(ids=ids))