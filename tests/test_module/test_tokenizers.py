"""Testing tokenizers in the nlp.py module.
"""
from florabert import config, nlp


def test_dnabert_tokenizer():
    ex_seq = "AAATCGTCGCGGGCGCTCGCTATATATCGGCTAGCTAACTCGCCCG"
    tokenizer = nlp.DNABERTTokenizer.from_pretrained(
        config.models / "dnabert" / "tokenizer", k=6, max_len=512
    )
    tokenized = tokenizer(ex_seq)
    decoded = tokenizer.decode(ex_seq["input_ids"])

    assert (
        ex_seq == decoded
    ), f"Input ({ex_seq}) does not match decoded sequence ({decoded})."
