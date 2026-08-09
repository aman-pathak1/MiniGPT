from src.tokenizer import GPTTokenizer


tokenizer = GPTTokenizer()


text = "Hello! Umm... 😊 I am Aman."


tokens = tokenizer.encode(text)


decoded_text = tokenizer.decode(tokens)


print("=" * 50)
print("MiniGPT Tokenizer Test")
print("=" * 50)


print("\nOriginal Text:")
print(text)


print("\nToken IDs:")
print(tokens)


print("\nDecoded Text:")
print(decoded_text)


print("\nVocabulary Size:")
print(tokenizer.vocab_size)


# ============================================================
# Encode Test
# ============================================================

assert isinstance(tokens, list), (
    "Tokenizer encode() must return a list"
)


assert len(tokens) > 0, (
    "Tokenizer returned no tokens"
)


# ============================================================
# Decode Test
# ============================================================

assert isinstance(decoded_text, str), (
    "Tokenizer decode() must return a string"
)


assert decoded_text == text, (
    "Decoded text does not match original text"
)


# ============================================================
# Vocabulary Test
# ============================================================

assert tokenizer.vocab_size == 50257, (
    f"Expected GPT-2 vocabulary size 50257, "
    f"got {tokenizer.vocab_size}"
)


# ============================================================
# Token ID Range Test
# ============================================================

assert all(
    0 <= token_id < tokenizer.vocab_size
    for token_id in tokens
), "Token ID is outside vocabulary range"


print()
print("=" * 50)
print("Tokenizer Test Passed Successfully!")
print("=" * 50)