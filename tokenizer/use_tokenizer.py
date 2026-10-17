
import os
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

text = "Once upon a little dragon, the dragon was happy."

encoding = tokenizer.encode(text)
token_ids = encoding.ids
decoded_text = tokenizer.decode(token_ids)

print("Token IDs:", token_ids)
print("Tokens:", encoding.tokens)
print("Decoded:", decoded_text)

assert decoded_text == text
print("Round-trip test passed.")
