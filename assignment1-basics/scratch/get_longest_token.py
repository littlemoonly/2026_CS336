import json

with open("cs336_basics/tinystories_bpe_vocab.json", "r") as f:
    vocab = json.load(f)

max_len = 0
max_str = []
for token_str, _ in vocab.items():
    if len(token_str) > max_len:
        max_len = len(token_str)
        max_str.clear()

    if len(token_str) == max_len:
        max_str.append(token_str)

print(f" max len = {max_len}")
for s in max_str:
    print(s, end=', ')