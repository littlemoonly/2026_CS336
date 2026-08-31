from tests.adapters import run_train_bpe

def test_train_bpe():
    input_path = "tests/aaa.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    # print(vocab)
    # print(merges)

if __name__ == '__main__':
    test_train_bpe()
