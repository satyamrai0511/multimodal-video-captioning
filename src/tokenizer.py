from transformers import AutoTokenizer

# Use a lightweight tokenizer (you can replace with others like GPT2 later)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(texts, max_length=32):
    """
    Tokenize a list of texts and return input_ids and attention_mask tensors.
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
