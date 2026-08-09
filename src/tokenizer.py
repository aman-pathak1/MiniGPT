import tiktoken


class GPTTokenizer:

    def __init__(self, encoding_name="gpt2"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str):
        return self.encoding.encode(text)

    def decode(self, token_ids):
        return self.encoding.decode(token_ids)

    @property
    def vocab_size(self):
        return self.encoding.n_vocab