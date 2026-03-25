
import os
import re
from collections import defaultdict

import regex

from cs336_basics.pretokenization_example import find_chunk_boundaries


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given corpus with GPT-2 style pre-tokenization.
    Uses find_chunk_boundaries to split the file into chunks at special token
    boundaries, avoiding loading the entire file into memory at once.

    Returns:
        vocab: mapping from token ID to token bytes
        merges: list of (bytes, bytes) merge pairs in order of creation
    """
    special_tokens = special_tokens or []
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 1. Split file into chunks at special token boundaries using find_chunk_boundaries
    #    Each chunk is guaranteed not to split a special token across boundaries.
    #    The special token used for splitting is the first one (or a dummy newline if none).
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    num_chunks = os.cpu_count() or 4

    word_freqs: dict[tuple[bytes, ...], int] = {}

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

            # 2. Split on special tokens so their bytes never participate in merges
            if special_tokens:
                sorted_tokens = sorted(special_tokens, key=len, reverse=True)
                pattern = "|".join(re.escape(t) for t in sorted_tokens)
                parts = re.split(pattern, chunk_text)
            else:
                parts = [chunk_text]

            # 3. Pre-tokenize (GPT-2 regex) and accumulate word frequencies
            for part in parts:
                if not part:
                    continue
                tokens = regex.findall(GPT2_PAT, part)
                for tok in tokens:
                    word = tuple(bytes([b]) for b in tok.encode("utf-8"))
                    word_freqs[word] = word_freqs.get(word, 0) + 1

    # 4. How many merges to perform
    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 5. Iterative BPE merging
    for _ in range(num_merges):
        # Count all adjacent pairs, weighted by word frequency
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for j in range(len(word) - 1):
                pair_counts[(word[j], word[j + 1])] += freq

        if not pair_counts:
            break

        # Find the most frequent pair (break ties by lexicographic order of pair)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)

        # Merge best_pair in every word
        new_word_freqs: dict[tuple[bytes, ...], int] = {}
        for word, freq in word_freqs.items():
            new_word: list[bytes] = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                    new_word.append(word[j] + word[j + 1])
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            key = tuple(new_word)
            new_word_freqs[key] = new_word_freqs.get(key, 0) + freq
        word_freqs = new_word_freqs

    # 6. Build vocab: 256 base bytes → special tokens → merged tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    for i, (a, b) in enumerate(merges):
        vocab[256 + len(special_tokens) + i] = a + b

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],merges: list[tuple[bytes, bytes]],special_tokens: list[str] | None = None,):
        self.vocab = vocab
        # print(self.vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}


    def _split_with_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        sorted_tokens = sorted(self.special_tokens, key = len, reverse=True)

        pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"

        parts = re.split(pattern=pattern, string = text);

        return [p for p in parts if p]
    
    def _encode_chunk(self, part: str) -> list[int]:
        # 将 bs 中的每个 byte，都转成 bytes， 方便后续 merge 时的查找和比较
        bs = part.encode("utf-8")
        tokens = [bytes([b]) for b in bs] 
        # print(f"初始: {tokens}")   # [b'\xc3', b'\xa9']

        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            min_pair = None
            min_priority = float("inf")
            for pair in pairs:
                if pair in self.merge_priority:
                    if self.merge_priority[pair] < min_priority:
                        min_priority = self.merge_priority[pair]
                        min_pair = pair

            if min_pair is None:
                break;
            # print(f"最佳 pair: {min_pair}, tokens: {tokens}")

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == min_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2      # ✅ 跳过两个
                else:
                    new_tokens.append(tokens[i])
                    i += 1      # ✅ 正常前进
            tokens = new_tokens

        # 这套规则下，保证了，没有 bytes to id 不认识的输入
        ids = [self.bytes_to_id[token] for token in tokens]
        # print(tokens)
        # print(ids)
        return ids
        

    def encode(self, text: str) -> list[int]:
        """对一个 token 列表，反复合并优先级最高的 pair"""

        txts: list[str] = self._split_with_special_tokens(text)
        GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        idx: list[int] =[]
        for t in txts:
            if t in self.special_tokens:
                idx.append(self.bytes_to_id[t.encode("utf-8")])
            else:
                ts = regex.findall(GPT2_PAT, t)
                for chunk in ts:
                    idx += self._encode_chunk(chunk)
        return idx;
    
    def encode_iterable(self, iterable):
        """逐行读取，逐个 yield token id，避免一次性加载整个文本到内存"""
        for line in iterable:
            for id in self.encode(line):
                yield id

    def decode(self, token_ids: list[int]) -> str:
        # print(type(self.vocab[token_ids[0]]))
        # j = [self.vocab[id] for id in token_ids]
        # print(j)

        return b"".join(self.vocab[id] for id in token_ids).decode("utf-8", errors="replace")

