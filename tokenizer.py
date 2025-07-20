# build_tokenizer.py

import pickle
from pathlib import Path

# Load your training data file (the same used for training)
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character-level mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Save tokenizer mappings
tokenizer_data = {
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
}

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer_data, f)

print("✅ Tokenizer saved to tokenizer.pkl")
