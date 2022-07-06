""" Utilities for reading and writing data files.
"""
import csv
import itertools
import multiprocessing as mp
import os
import random
from pathlib import PosixPath
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio import Seq, SeqIO
from datasets import load_dataset
from sklearn.base import BaseEstimator
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    default_data_collator,
)

from . import config, utils

# To avoid huggingface warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
UBUNTU_ROOT = str(config.root)


def load_seqs(files: Union[list, str, PosixPath], pbar=True) -> List[Seq.Seq]:
    """Load all sequences from `files` (FASTA) into a list."""
    files = utils.ensure_iterable(files)
    res = []
    iterator = tqdm(list(files)) if pbar else files
    for fname in iterator:
        with open(fname, "r") as f:
            seqs = SeqIO.parse(f, "fasta")
            for seq in seqs:
                res.append(seq)
    return res


def load_csvs(
    files: Union[list, str, PosixPath],
    show_pbar=False,
    notebook=False,
    subset=None,
    add_doc_ids: bool = False,
    nrows: int = None,
) -> List[tuple]:
    """Load all csv files in `files` into a list of tuples."""
    res = []
    files = utils.ensure_iterable(files)
    if subset is not None:
        subset = utils.ensure_iterable(subset)

    doc_id = 0
    n_processed = 0
    for f in files:
        with open(f, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            if show_pbar:
                n_lines = utils.count_lines(f) - 1
                total = nrows if nrows else n_lines
                pbar = tqdm_notebook(total=total) if notebook else tqdm(total=total)

            for row in reader:
                if nrows:
                    if n_processed > nrows:
                        break
                data = []
                if add_doc_ids:
                    data.append(doc_id)
                if subset is not None:
                    if len(subset) == 1:
                        data.append(row[subset[0]])
                    else:
                        data.extend([row[k] for k in subset])
                else:
                    data.extend(row.values())
                if len(data) == 1:
                    data = data[0]
                else:
                    data = tuple(data)
                res.append(data)
                if show_pbar:
                    pbar.update()

                doc_id += 1
                n_processed += 1

            if show_pbar:
                pbar.close()

            if n_processed > total:
                break

    return res


def load_b73_seq_data(
    n_sample: int = None, indices: bool = True, min_seq_len: int = None
):
    df = pd.read_csv(
        config.data_final / "Zmb73" / "merged_seq_data_b73.csv", index_col=0
    )
    if n_sample:
        df = df.sample(n=n_sample).reset_index(drop=True)

    if min_seq_len:
        df = df[df["seq"].apply(lambda x: len(x) >= min_seq_len)]

    if indices:
        seq_data = list(zip(df.index.tolist(), df["seq"].tolist()))
        seq_names = list(zip(df.index.tolist(), df["name"].tolist()))
    else:
        seq_data = df["seq"].tolist()
        seq_names = df["name"].tolist()

    return seq_data, seq_names


def load_b73_genex_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test split of genex data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: df_train, df_test
    """
    genex_data = pd.read_csv(
        config.data_final / "Zmb73" / "merged_genex_data_b73.csv", index_col=0
    ).reset_index(drop=True)
    # Capping top gene expression level at 100
    # genex_data['gene_expression_level'] = genex_data['gene_expression_level'].apply(lambda x: min(x, 100))
    labels = pd.read_csv(config.data_final / "Zmb73" / "genex_train_test_label.csv")
    df_train = utils.genex_long_to_wide(genex_data[labels["train_test"] == "train"])
    df_test = utils.genex_long_to_wide(genex_data[labels["train_test"] == "test"])
    return df_train, df_test


def load_datasets(
    tokenizer: PreTrainedTokenizer,
    train_data: Union[str, PosixPath],
    eval_data: Optional[Union[str, PosixPath]] = None,
    test_data: Union[str, PosixPath] = None,
    file_type: str = "csv",
    delimiter: str = "\t",
    seq_key: str = "sequence",
    shuffle: bool = True,
    filter_empty: bool = False,
    min_seq_len: int = None,
    transformation: str = None,
    log_offset: Union[float, int] = 1,
    preprocessor: BaseEstimator = None,
    tissue_subset: Union[str, int, list] = None,
    nshards: int = None,
    threshold: float = None,
    discretize: bool = False,
    kmer: int = None,
    n_workers: int = mp.cpu_count(),
    position_buckets: Tuple[int] = None,
    random_seed: int = None,
    **kwargs,
) -> Dataset:
    """Load and cache data using Huggingface datasets library

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer to apply to the sequences
        train_data (Union[str, PosixPath]): location of training data
        eval_data (Union[str, PosixPath], optional): location of evaluation data. Defaults to None.
        test_data (Union[str, PosixPath], optional): location of test data. Defaults to None.
        file_type (str, optional): type of file. Possible values are 'text' and 'csv'. Defaults to 'csv'.
        delimiter (str, optional): Defaults to '\t'.
        seq_key (str, optional): Column name of sequence data Can be 'sequence', 'seq', or 'text'. Defaults to 'sequence'.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        filter_empty (bool, optional): Whether to filter out empty sequences. Defaults to False.
            NOTE: This completes an additional iteration, which can be time-consuming.
            Only enable if you have reason to believe that preprocessing steps will
            result in empty sequences.
        transformation (str, optional): type of transformation to apply.
            Options are 'log', 'boxcox'. Defaults to None.
        log_offset (Union[float, int]): value to offset gene expression values
            by before log transforming. Defaults to 1.
        preprocessor (BaseEstimator): preprocessor Yeoh-Johnson transformation.
        tissue_subset (Union[str, int, list], optional): tissues to subset labels to.
            Defaults to None.
        nshards (int, optional): Number of shards to divide data into, only
            keeping the first. Defaults to None.
        threshold (float, optional): filter out rows where all labels are
            below `threshold`. OR if `discretize` is True, see `discretize`.
            Defaults to None.
        discretize (bool, optional): set gene expression values below
            `threshold` to 0, above `threshold` to 1.
        kmer (int, optional): whether to run the kmer flip experiment and if so,
            how large kmers to flip. Defaults to None.
        n_workers (int, optional): number of processes to use for preprocessing.
            Defaults to `mp.cpu_count()` (number of available CPUs).
        position_buckets (Tuple[int], optional): the different buckets for the bucketed
            positional importance experiment

    Returns:
        Dataset
    """
    data_files = {"train": str(train_data)}
    if eval_data:
        data_files["eval"] = str(eval_data)
    if test_data:
        data_files["test"] = str(test_data)
    if file_type == "csv":
        kwargs.update({"delimiter": delimiter})
    datasets = load_dataset(file_type, data_files=data_files, **kwargs)

    if min_seq_len is not None:
        print(f"Filtering sequences with length below {min_seq_len}")
        filter_min_length = make_min_length_filter(min_seq_len, seq_key=seq_key)
        datasets = datasets.filter(filter_min_length)

    if nshards:
        print(f"Keeping 1/{nshards} of the dataset")
        for key in ["train", "eval", "test"]:
            if key in datasets:
                datasets[key] = datasets[key].shard(nshards, 0)
    if kmer is not None:
        if position_buckets is not None:
            print("Performing bucketed kmer flip experiment")
        else:
            print("Performing kmer flip experiment")
        kmer_flip = make_kmer_flip_function(
            seq_key, kmer, buckets=position_buckets, random_seed=random_seed
        )
        datasets = datasets.map(kmer_flip, batched=True, num_proc=n_workers)

    # Tokenizing
    preprocess_fn = make_preprocess_function(tokenizer, seq_key=seq_key)
    print("Tokenizing")
    datasets = datasets.map(preprocess_fn, batched=True, num_proc=n_workers)
    if filter_empty:
        datasets = datasets.filter(filter_empty_sequence)

    if file_type != "text":
        datasets = datasets.map(
            utils.convert_str_to_tnsr, batched=True, num_proc=n_workers
        )
        if tissue_subset is not None:
            print(f"Subsetting to tissues {tissue_subset}")
            datasets = datasets.map(
                lambda x: subset_tissues(x, tissue_subset),
                batched=True,
                num_proc=n_workers,
            )
        if discretize:
            assert (
                threshold is not None
            ), "if `discretize` is True, must supply `threshold`."
            print(f"Discretizing with threshold {threshold}")
            datasets = datasets.map(
                lambda x: discretize_genex_values(x, threshold),
                batched=True,
                num_proc=None,
            )
        else:
            if threshold:
                print(f"Dropping gene expression below {threshold}")
                datasets = datasets.filter(
                    lambda x: drop_below_threshold(x, threshold), num_proc=1
                )
            if transformation == "custom":
                assert (
                    preprocessor is not None
                ), "must supply preprocessor if specifying custom transformation."
                print("Custom transformation")
                datasets = datasets.map(
                    lambda x: preprocess_custom(x, preprocessor),
                    batched=True,
                    num_proc=1,
                )
            elif transformation == "log":
                log_offset = log_offset or 0
                print(f"Log transformation with offset {log_offset}")
                datasets = datasets.map(
                    lambda x: preprocess_log_transform(x, log_offset),
                    batched=True,
                    num_proc=n_workers,
                )
    if shuffle:
        seed = config.settings["random_seed"]
        datasets = datasets.shuffle(seeds={"train": seed, "eval": seed, "test": seed})
    return datasets


def make_preprocess_function(tokenizer, seq_key: str = "sequence") -> callable:
    """Make a preprocessing function that selects the appropriate column and
    tokenizes it.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer to apply to each sequence
        seq_key (str, optional): column name of the text data. Defaults to 'sequence'.

    Returns:
        callable: preprocessing function
    """

    def preprocess_function(examples):
        if seq_key:
            seqs = examples[seq_key]
        else:
            seqs = examples
        return tokenizer(
            seqs,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
        )

    return preprocess_function


# function to flip a random k-mer in the sequence. The output seq size is the same as the original seq size
def make_kmer_flip_function(
    seq_key: Dict[str, List[str]],
    kmer: int,
    buckets: Tuple[int] = None,
    random_seed: int = None,
):
    """return a function that will perform random k-mer flip
    Args:
        seq_key (str, optional): Column name of sequence data Can be 'sequence', 'seq', or 'text'. Defaults to 'sequence'.
        kmer (int): numebr k for the k-mer
        buckets (Tuple[int, optional]): buckets if performing a bucketed kmer flip

    Returns:
        callable: k-mer flip function
    """
    bases = ["A", "T", "G", "C"]
    kmer_list = ["".join(p) for p in itertools.product(bases, repeat=kmer)]
    maxlen = 1000
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random.Random()

    def kmer_flip(examples: list):
        # Do the kmer flipping with a batch of exmaples
        if seq_key:
            seqs = examples[seq_key]
        else:
            seqs = examples
        flipped_index = []
        temp_seqs = seqs.copy()
        kmers_to = []
        kmers_from = []
        for i, seq in enumerate(seqs):
            seq = list(seq)
            # Choose an index to flip
            if buckets is not None:
                # first choose a bucket, then an index to flip
                buckets_for_seq = [(b1, b2) for b1, b2 in buckets if b2 <= len(seq)]
                if len(buckets_for_seq) == 0:
                    bucket_start, bucket_end = 0, len(seq)
                else:
                    bucket_start, bucket_end = rng.choice(buckets_for_seq)
                flip_num = rng.randint(bucket_start, bucket_end - kmer)
            else:
                flip_num = rng.randint(0, min(maxlen, len(seq)) - kmer)
            flipped_index.append(flip_num)
            kmer_to = rng.choice(kmer_list)
            # kmer_to = np.random.choice(kmer_list, 1)[0]
            kmer_from = "".join(seq[flip_num : flip_num + kmer])
            seq[flip_num : flip_num + kmer] = kmer_to
            temp_seqs[i] = "".join(seq)

            kmers_to.append(kmer_to)
            kmers_from.append(kmer_from)

            if len(kmer_from) < kmer:
                print(
                    f"{bucket_start=}, {flip_num=}, {bucket_end=}, {kmer_from=}, {len(seq)=}"
                )
        flipped_sequence = temp_seqs

        # Return a dict
        return {
            "sequence": flipped_sequence,
            "index": flipped_index,
            "kmer_to": kmers_to,
            "kmer_from": kmers_from,
        }

    # What you return will be added to the dataset, overriding existing fields that share the same keys
    return kmer_flip


def filter_empty_sequence(example: dict) -> bool:
    """Filter out empty sequences."""
    # sum(example['attention_mask']) gives the number of tokens, including SOS and EOS
    return sum(example["attention_mask"]) > 2


def make_min_length_filter(min_seq_len: int, seq_key: str = None) -> dict:
    def fn(example: dict) -> bool:
        if seq_key is not None:
            return len(example[seq_key]) >= min_seq_len
        return example >= min_seq_len

    return fn


def preprocess_log_transform(examples: dict, eps=1) -> dict:
    """Log transform values in a list, offsetting by `eps` (default 1) to avoid 0s"""
    log_transformed = []
    for ex in examples["labels"]:
        log_transformed.append([np.log(x + eps) for x in ex])
    return {"labels": log_transformed}


def preprocess_custom(examples: dict, preprocessor: BaseEstimator) -> dict:
    """Custom transformation based on sklearn transformer.

    Args:
        examples (dict): data to transform (with key `labels`)
        preprocessor (BaseEstimator): sklearn preprocessor

    Returns:
        dict: transformed examples
    """
    transformed = preprocessor.transform(examples["labels"])
    return {"labels": transformed}


def subset_tissues(
    examples: dict, subset: Union[str, int, List[Union[int, str]]]
) -> dict:
    """Subset the output labels  based on `subset`"""
    if isinstance(subset, (int, str)):
        subset = [subset]
    subset_idxs = []
    for tiss in subset:
        if isinstance(tiss, int):
            subset_idxs.append(tiss)
        else:
            subset_idxs.append(config.tissues.index(tiss))

    res = []
    for ex in examples["labels"]:
        res.append([ex[idx] for idx in subset_idxs])

    return {"labels": res}


def drop_below_threshold(example: dict, threshold: float) -> bool:
    """Drop values below `threshold`

    Args:
        example (dict): [description]
        threshold (float): [description]

    Returns:
        dict: [description]
    """
    return all([x > threshold for x in example["labels"]])


def discretize_genex_values(examples: dict, threshold: float) -> dict:
    res = []
    for ex in examples["labels"]:
        res.append([int(x >= threshold) for x in ex])
    return {"labels": res}


# TODO: Write this class
class DataCollatorForDNABERT(DataCollatorForLanguageModeling):
    pass


def load_data_collator(model_type: str, tokenizer=None, mlm_prob=None):
    if model_type == "language-model":
        assert (
            tokenizer is not None
        ), "tokenizer must not be None if model is type language-model"
        assert (
            mlm_prob is not None
        ), "mlm_prob must not be None if model is type language-model"

        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob
        )
    elif model_type == "dnabert-lm":
        return DataCollatorForDNABERT(mlm_prob=mlm_prob)
    else:
        return default_data_collator
