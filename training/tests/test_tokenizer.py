import os
import json
from collections import Counter

def test_tokenizer_roundtrip(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    detok = tokenizer.decode(token_ids)
    
    print("Original:", text)
    print("Tokens:", tokens)
    print("Detokenized:", detok)
    
    if text != detok:
        print("Warning: Tokenizer detokenization mismatch!")
    else:
        print("Tokenizer works perfectly on this input.")
    print()

def check_special_chars(tokenizer):
    special_chars = [":", ";", "-", " ", "_", ".", "<s>", "<pad>"]
    vocab = tokenizer.get_vocab()

    print("Checking special characters in tokenizer vocabulary...")
    for ch in special_chars:
        if ch not in vocab:
            print(f"Warning: character '{ch}' not in tokenizer vocab!")
    for digit in "0123456789":
        if digit not in vocab:
            print(f"Warning: digit '{digit}' not in tokenizer vocab!")
    print()

def check_oov_tokens(dataset, tokenizer):
    unk_token = tokenizer.unk_token
    unk_count = 0
    total_tokens = 0
    for item in dataset:
        label_str = "; ".join(f"{k}: {v}" for k, v in item.items())
        tokens = tokenizer.tokenize(label_str)
        total_tokens += len(tokens)
        unk_count += tokens.count(unk_token)
    print(f"Total tokens: {total_tokens}")
    print(f"Unknown tokens (OOV): {unk_count}")
    print(f"Percentage unknown: {100 * unk_count / total_tokens:.2f}%")
    print()

if __name__ == "__main__":
    # Import your tokenizer / processor here
    from transformers import DonutProcessor

    # Load your pretrained processor/tokenizer, e.g.:
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    tokenizer = processor.tokenizer

    # Test 1: Roundtrip tokenization/detokenization on sample text
    sample_text = '{"name": "Lori Larson", "dob": "1955-09-30", "id_number": "526756021", "expiry": "2025-08-16"}'
    test_tokenizer_roundtrip(tokenizer, sample_text)

    # Test 2: Check special characters and digits in vocab
    check_special_chars(tokenizer)

    # Test 3: Check OOV tokens in your dataset
    # Load your dataset JSON labels here:
    # Replace this with your actual path to label JSON files
    label_dir = "data/train"

    dataset = []
    for fname in os.listdir(label_dir):
        if fname.endswith(".json"):
            with open(os.path.join(label_dir, fname), "r") as f:
                data = json.load(f)
                dataset.append(data)

    check_oov_tokens(dataset, tokenizer)
