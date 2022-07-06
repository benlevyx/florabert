"""Utility and modules for NLP preprocessing
"""
import re
import typing
import itertools as it
from pathlib import PosixPath

import tokenizers as tk
import torch
from transformers import BertTokenizerFast
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import bpe_tokenizer


SOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3
MASK_IDX = 4
SPECIAL_TOKENS = {
    "<SOS>": SOS_IDX,
    "<EOS>": EOS_IDX,
    "<PAD>": PAD_IDX,
    "<UNK>": UNK_IDX,
    "<MASK>": MASK_IDX,
}
NUCLEOTIDES = ["A", "T", "C", "G", "N"]
BASE_VOCAB = NUCLEOTIDES + list(SPECIAL_TOKENS.keys())

BERT_SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


class Tokenizer:
    """Base Class for tokenization of DNA sequence into motifs and unstructured
    nucleotides.
    """

    def __init__(self, vocab):
        self.fitted = False
        self.vocab2idx = SPECIAL_TOKENS.copy()
        self.vocab2idx.update({v: i + len(SPECIAL_TOKENS) for i, v in enumerate(vocab)})
        self.idx2vocab = {i: v for v, i in self.vocab2idx.items()}
        self.vocab = list(self.vocab2idx.keys())
        self.vocab_size = len(self.vocab)
        # Ensuring lower case covered
        for nt in self.vocab:
            self.vocab2idx[nt.lower()] = self.vocab2idx[nt]

    def fit(self, corpus=None):
        self.fitted = True
        return self

    def transform(self, corpus):
        res = []
        for doc in corpus:
            res.append(self.tokenize(doc))
        return res

    def tokenize(self, doc):
        raise NotImplementedError()

    def decode(self, doc):
        res = []
        for idx in doc:
            if type(idx) == torch.Tensor:
                idx = idx.item()
            res.append(self.idx2vocab.get(idx, "<UNK>"))
        return "".join(res)

    def encode(self, doc):
        # For compatibility with Huggingface API
        return self.tokenize(doc)

    def __call__(self, doc):
        return self.tokenize(doc)

    def get_vocab(self) -> dict:
        return self.vocab2idx.copy()

    def get_vocab_size(self) -> int:
        return self.vocab_size


class NGramTokenizer(Tokenizer):
    """Generalized unigram tokenizer that recognizes all possible n-mers
    as individual units.
    """

    def __init__(self, n=1, vocab=None, offset=0, overlap=False):
        if not vocab:
            vocab = ["".join(e) for e in it.product(NUCLEOTIDES, repeat=n)]
        self.n = n
        self.offset = offset
        self.overlap = overlap
        super().__init__(vocab)

    def tokenize(self, doc):
        step = 1 if self.overlap else self.n
        return [
            self.vocab2idx.get(doc[i : i + self.n], UNK_IDX)
            for i in range(self.offset, len(doc) - (self.n - 1), step)
        ]


class DNABERTPreTokenizer:
    def __init__(self, k, max_len):
        self.k = k

    def tokenize_seqs(self, seqs):
        return [self.seq2kmer(seq) for seq in seqs]

    def seq2kmer(self, seq):
        """
        Convert original sequence to kmers

        From https://github.com/jerryji1993/DNABERT/blob/master/motif/motif_utils.py

        Arguments:
        seq -- str, original sequence.
        k -- int, kmer of length k specified.

        Returns:
        kmers -- str, kmers separated by space
        """
        kmer = [seq[x : x + self.k] for x in range(len(seq) + 1 - self.k)]
        kmers = " ".join(kmer)
        return kmers

    def __call__(self, seqs):
        return self.tokenize_seqs(seqs)


def load_bpe_tokenizer(
    save_path: typing.Union[str, PosixPath, None] = None,
    add_special_tokens: bool = False,
) -> tk.Tokenizer:
    """Load a HuggingFace BPE tokenizer from `save_path`.

    Args:
        save_path (Union[str, PosixPath]): Path to trained tokenizer json file
        add_special_tokens (bool)

    Returns:
        (tk.Tokenizer): Pretrained BPE tokenizer
    """
    save_path = save_path or bpe_tokenizer
    tokenizer = tk.Tokenizer.from_file(str(save_path))
    if add_special_tokens:
        tokenizer.add_special_tokens(BERT_SPECIAL_TOKENS)

    return tokenizer


class DNABERTTokenizer:
    def __init__(self, *args, k=6, **kwargs):
        """Tokenizer for DNABERT

        Has-a BERTTokenizerFast

        Args:
            k (int, optional): Size of overlapping kmers. Defaults to 6.
        """
        self.k = k
        self.tokenizer = BertTokenizerFast.from_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DNABERTTokenizer(*args, **kwargs)

    @property
    def model_max_length(self):
        return self.tokenizer.model_max_length

    @staticmethod
    def _seq2kmer(seq, k):
        """
        Convert original sequence to kmers

        From https://github.com/jerryji1993/DNABERT/blob/master/motif/motif_utils.py

        Arguments:
        seq -- str, original sequence.
        k -- int, kmer of length k specified.

        Returns:
        kmers -- str, kmers separated by space
        """
        kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
        kmers = " ".join(kmer)
        return kmers

    @staticmethod
    def _kmer2seq(kmers: str):
        """
        Convert kmers to original sequence

        From https://github.com/jerryji1993/DNABERT/blob/master/motif/motif_utils.py

        Arguments:
        kmers -- str, kmers separated by space.

        Returns:
        seq -- str, original sequence.
        """
        kmers = re.sub(r"\[CLS\] | \[SEP\]", "", kmers)
        kmers_list = kmers.split(" ")
        bases = [kmer[0] for kmer in kmers_list[0:-1]]
        bases.append(kmers_list[-1])
        seq = "".join(bases)
        assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
        return seq

    @staticmethod
    def _call_possible_iterable(
        fn: callable, inputs: typing.Union[tuple, list, str], *args, **kwargs
    ):
        return_list = True
        if type(inputs) not in [list, tuple]:
            inputs = [inputs]
            return_list = False

        res = [fn(item, *args, **kwargs) for item in inputs]
        if return_list:
            return res
        else:
            return res[0]

    def _call_seq2kmer(self, method, text, *args, **kwargs):
        kmers = self._call_possible_iterable(self._seq2kmer, text, self.k)
        # kmers = self._seq2kmer(text, self.k)
        return method(kmers, *args, **kwargs)

    def _call_kmer2seq(self, method, ids, *args, **kwargs):
        kmers = method(ids, *args, **kwargs)
        return self._call_possible_iterable(self._kmer2seq, kmers)
        # return self._kmer2seq(kmers)

    def encode(self, text, *args, **kwargs):
        return self._call_seq2kmer(self.tokenizer.encode, text, *args, **kwargs)

    def batch_encode(self, text, *args, **kwargs):
        return self._call_seq2kmer(self.tokenizer.batch_encode, text, *args, **kwargs)

    def encode_plus(self, text, *args, **kwargs):
        return self._call_seq2kmer(self.tokenizer.encode_plus, text, *args, **kwargs)

    def batch_encode_plus(self, batch, *args, **kwargs):
        return [
            self._call_seq2kmer(self.tokenizer.batch_encode_plus, text, *args, **kwargs)
            for text in batch
        ]

    def decode(self, ids, *args, **kwargs):
        return self._call_kmer2seq(self.tokenizer.decode, ids, *args, **kwargs)

    def batch_decode(self, batch, *args, **kwargs):
        return [
            self._call_kmer2seq(self.tokenizer.batch_decode, ids, *args, **kwargs)
            for ids in batch
        ]

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, text, *args, **kwargs):
        return self._call_seq2kmer(self.tokenizer, text, *args, **kwargs)


def get_kmer_counts(X: pd.Series, k: int) -> typing.Tuple[np.ndarray, typing.List[str]]:
    """
    It takes a training and test set of DNA sequences, and returns a bag-of-words representation of the
    sequences, where the words are all possible k-mers

    Args:
      X (pd.Series): the data
      k (int): the length of the k-mer

    Returns:
      the kmer counts for the training and test data.
    """
    kmers = ["".join(i) for i in list(it.product(NUCLEOTIDES, repeat=k))]
    X_bow = np.zeros((len(X), len(kmers)), dtype=int)
    for i, j in tqdm(list(enumerate(kmers))):
        X_bow[:, i] = X.fillna("").str.count(j)
    return X_bow, kmers
