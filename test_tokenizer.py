from src.tokenizer import GPTTokenizer

tokenizer = GPTTokenizer()

text = "Hello! Umm... 😊 I am Aman."

tokens = tokenizer.encode(text)

print("Original Text:")
print(text)

print("\nToken IDs:")
print(tokens)

print("\nDecoded Text:")
print(tokenizer.decode(tokens))

print("\nVocabulary Size:")
print(tokenizer.vocab_size)