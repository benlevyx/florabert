"""
    Tests the trained BPE tokenizer from scripts/0-data-loading-processing/06-BPE-encode.py
"""
from florabert import config
from tokenizers import Tokenizer


def test_tokenizer():
    # Load the tokenizer
    tokenizer_path = config.data_final / "tokenizer"
    save_path = str(tokenizer_path / "maize_bpe_full.tokenizer.json")
    tokenizer = Tokenizer.from_file(save_path)

    # Basic statistics
    vocab = tokenizer.get_vocab()
    assert sum([c in vocab for c in "ATCG"]) == 4, "One of 'ATCG' is not in the vocab"

    print("Total vocab size is:", tokenizer.get_vocab_size())
    print("First 10 vocabs:", [(k, vocab[k]) for k in list(vocab.keys())[:10]])

    # Test it on a particular sequence
    test = "CAGCCCGACCCCCTCACAGTCCAACGGTGCCCTGAGTTCGGCACCCTCTCTAGGAATAGAGAGGCTGCTCC\
CTCTGTACATGGGGGAGTTCTAATCTCCCCTATTTCGGTAATCTATGTTTTAACTGTAAAATGAATTCCTTTTTAGTAT\
AATTACCTGATAACAATATGTATTATGATACTACAAATATGGTAGTATTTTTTAGAACTCCAAAAACTGATGTAAAAAA\
GTCAAATAGCTCAGTTAAAGAGTAAATGGGAGCTGAATAGGGGGGAATGGTTGGAGTGGAGATGAAATATGGAGAATAA\
TAGTTGAGGGGGAGGTATTTAAATATGAATAGAAAGTACGAATGGAGGGATTTGAGAGAGGAATGGTTGAAGAGAATCT\
AAACTATTTTGAACCTTCTCTAGGTGTTGCTTAAATTTATAATCTGCGTAGCACTTTAGAAACATGTTTGCAGTTGCGA\
CATTGTCAAAATAACATTGATTGTCTAAAAATAAAGAAAATAACAGAGATAGTGTCTATGTCATGTGCACATAGACAAA\
GTTATTTTTGCAATGTTACACGTGTATCTTGACAATTATATCGACAACTTAATGGTACCGAGACATTTTGCCGTTTAAC\
AATGGATCATGTGATTTAGTATATGTTTGGTTTTAGAGACAATTAAGGATTATTCTTACTTGCATAACTTGTCATTCGA\
GAGAGCACGTACTTGTTCCATGTTCAATTCGATGTTGAGAAGGCTCAAGGCTTGCCTAGTTTATCTATTTTGCTCGTCG\
AAATCTAATTATTCATCCGGTGATTAATTTGTCCTTATACCTTTACAACAAAAATTCACTATCATCAACCTGTGCAAAG\
GAAATTGGAGGGAAAAATACATGTCATAGAATAAATATTTTCAGACAATTATTAGCACAAAAAATAACAAAGTTCGGTT\
TAATTTGCCCAATTTGTTGAGGTGAACTTCAGTAACAAAAGCAATCCCTACATTTTCTGG"
    encoded = tokenizer.encode(test)
    encoded_tokens = encoded.tokens
    print(encoded)
    print("First 10 tokens:", encoded_tokens[:10])

    # Try decode
    decoded = tokenizer.decode(encoded.ids)
    assert (
        decoded == test
    ), f"Decoded sequence is not the same as the original sequence!\
                              test: {test}\n\
                              decoded: {decoded}"


if __name__ == "__main__":
    test_tokenizer()
