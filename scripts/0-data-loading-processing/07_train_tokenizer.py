""" Training byte-level BPE for RoBERTa model
"""
from tokenizers import ByteLevelBPETokenizer

from florabert import config


SETTINGS = config.settings["transformer"]["tokenizer"]
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
if not TOKENIZER_DIR.exists():
    TOKENIZER_DIR.mkdir()
DATA_DIR = config.data_final / "transformer" / "seq"
TRAIN_DATA = DATA_DIR / "all_seqs_train_sample.txt"
SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]


def main():
    tokenizer = ByteLevelBPETokenizer()
    print("Training tokenizer")
    tokenizer.train(
        files=str(TRAIN_DATA),
        vocab_size=SETTINGS["vocab_size"] + len(SPECIAL_TOKENS),
        special_tokens=SPECIAL_TOKENS,
    )
    print("Saving tokenizer")
    tokenizer.save_model(str(TOKENIZER_DIR))
    print("Done")


if __name__ == "__main__":
    main()
