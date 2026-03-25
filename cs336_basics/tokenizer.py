
import re
import regex 
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

