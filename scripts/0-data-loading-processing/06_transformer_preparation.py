"""
[DEPRECATED]

Data preparation for CornBERT transformer model

OUTPUT:
    1. seqs_tokenized_6gram_all.txt
    2. seqs_tokenized_6gram_zmb73.txt
"""
from types import GeneratorType
from typing import Union
from pathlib import PosixPath
from csv import DictReader
import json
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from inari import config, utils, nlp


SETTINGS = config.settings['transformer']['data']
SEQ_DIR = config.data_processed / 'combined'
INFILE_B73 = config.data_final / 'Zmb73' / 'merged_seq_data_b73.csv'
INFILE_B73_GENEX = config.data_final / 'Zmb73' / 'B73_genex.txt'
B73_TRAIN_TEST_LABELS = config.data_final / 'Zmb73' / 'newB73_genex_train_test_label.csv'

OUTDIR = config.data_final / 'transformer'
OUTDIR_SEQ = OUTDIR / 'seq'
OUTDIR_GENEX = OUTDIR / 'genex'
for d in [OUTDIR_SEQ, OUTDIR_GENEX]:
    d.mkdir(exist_ok=True, parents=True)

OUTFILE_ALL_SEQS_TRAIN = OUTDIR_SEQ / 'all_seqs_train.txt'
OUTFILE_ALL_SEQS_TEST = OUTDIR_SEQ / 'all_seqs_test.txt'
OUTFILE_ALL_SEQS_TRAIN_SAMP = OUTDIR_SEQ / 'all_seqs_train_sample.txt'
OUTFILE_ALL_SEQS_TEST_SAMP = OUTDIR_SEQ / 'all_seqs_test_sample.txt'

OUTFILE_MAIZE_TRAIN = OUTDIR_SEQ / 'maize_seqs_train.txt'
OUTFILE_MAIZE_TEST = OUTDIR_SEQ / 'maize_seqs_test.txt'

OUTFILE_B73_TRAIN_GENEX = OUTDIR_GENEX / 'train.tsv'
OUTFILE_B73_TEST_GENEX = OUTDIR_GENEX / 'dev.tsv'

OUTDIR_TOKENIZER = config.models / 'bpe-tokenizer'
OUTDIR_TOKENIZER.mkdir(parents=True, exist_ok=True)


def process_seq(seq: str, max_seq_len: int = None) -> str:
    """Perform any necessary processing of individual sequences.
    Here, just truncate to `max_seq_len` from the left.

    Args:
        seq (str): Sequence to be processed
        max_seq_len (int, optional): Maximum length of a sequence,
            counting from the right. Defaults to None.

    Returns:
        str: [description]
    """
    if max_seq_len:
        return seq[-max_seq_len:]
    else:
        return seq


def process_seq_file(infile: PosixPath, outfile_train: PosixPath, outfile_test: PosixPath,
                     seq_key='sequence', test_size: float = 0.2, sample_prob: float = None,
                     outfile_train_sample: PosixPath = None, outfile_test_sample: PosixPath = None,
                     **process_seq_kwargs):
    """Process the sequence (text) files into a format for the transformer model

    Args:
        infile (PosixPath): Path to sequence data (csv)
        outfile_train (PosixPath): The path to the training data
        outfile_test (PosixPath): The path to the test data
        seq_key (str, optional): The column name for the raw sequences in `infile`. Defaults to 'sequence'.
        test_size (float, optional): Proportion of sequences to hold out as test. Defaults to 0.2.
        sample_prob (float, optional): Proportion of sequences to sample for a lightweight dataset. Defaults to None.
        outfile_train_sample (PosixPath, optional): File for the lightweight training data. Defaults to None.
        outfile_test_sample (PosixPath, optional): File for the lightweight test data. Defaults to None.
    """
    open_files = []
    if sample_prob:
        if outfile_train_sample:
            fout_train_samp = outfile_train_sample.open('a')
            open_files.append(fout_train_samp)
        if outfile_test_sample:
            fout_test_samp = outfile_test_sample.open('a')
            open_files.append(fout_test_samp)

    fin = infile.open('r', newline='')
    fout_train = outfile_train.open('a')
    fout_test = outfile_test.open('a')
    open_files.extend([
        fin,
        fout_train,
        fout_test
    ])

    reader = DictReader(fin)
    n_rows = utils.count_lines(infile)
    with tqdm(total=n_rows) as pbar:
        for i, row in enumerate(reader):
            seq = process_seq(row[seq_key], **process_seq_kwargs)
            if test_size:
                if np.random.binomial(1, test_size) == 1:
                    fout_test.write(seq + '\n')
                    if sample_prob:
                        if np.random.binomial(1, sample_prob) == 1:
                            fout_test_samp.write(seq + '\n')
                else:
                    fout_train.write(seq + '\n')
                    if sample_prob:
                        if np.random.binomial(1, sample_prob) == 1:
                            fout_train_samp.write(seq + '\n')
            else:
                fout_train.write(seq + '\n')
                if sample_prob:
                    if np.random.binomial(1, sample_prob) == 1:
                        fout_train_samp.write(seq + '\n')
            pbar.update()
    for f in open_files:
        f.close()


def process_genex_data(seqs_file: PosixPath, genex_file: PosixPath,
                       train_test_labels_file: PosixPath,
                       outfile_train: PosixPath, outfile_test: PosixPath,
                       genex_tissue: Union[None, str] = None,
                       genex_transform: str = 'log',
                       log_offset: float = 1.,
                       categorical_threshold: float = 1.,
                       **process_seq_kwargs):
    """Process the gene expression data for the transformer model

    Args:
        seqs_file (PosixPath): Path to the raw sequence file
        genex_file (PosixPath): Path to the gene expression data
        train_test_labels_file (PosixPath): Path to file containing
            'train' or 'test' for each row
        outfile_train (PosixPath): Path to the output file for the training data
        outfile_test (PosixPath): Path to the output file for the testing data
        genex_tissue (Union[None, str], optional): Tissue to select. Defaults to None.
        genex_transform (str, optional): Transformation to perform on gene
            expression values. Defaults to log(x + `log_offset`) ('log'). Options are
            {'log', 'identity', 'categorical'}
        log_offset (float, optional): Amount to offset values before applying natural log.
            Defaults to 1.
        categorical_threshold (float, optional): Threshold for turning gene
            expression into binary expressed/not expressed variable.
            Defaults to 1.
    """
    df_seqs = pd.read_csv(seqs_file, index_col=0)
    df_genex = pd.read_csv(genex_file, sep='\t', index_col=0)
    df_labels = pd.read_csv(train_test_labels_file)
    df_labels.index = df_genex.index

    df_seqs['name'] = df_seqs['name'].apply(lambda x: x.split('::')[0])
    df_seqs = df_seqs.set_index('name')
    df_seqs['seq'] = df_seqs['seq'].apply(process_seq, **process_seq_kwargs)

    if genex_tissue:
        df_genex = df_genex[[genex_tissue]]

    df_merged = pd.merge(df_seqs, df_genex, left_index=True, right_index=True, how='inner')
    for col in df_merged.columns:
        if col == 'seq':
            continue
        if genex_transform == 'identity':
            continue
        elif genex_transform == 'log':
            df_merged[col] = np.log(df_merged[col].values + log_offset)
        elif genex_transform == 'categorical':
            df_merged[col] = df_merged[col].apply(lambda x: int(x > categorical_threshold))
        else:
            raise ValueError(f"genex_transform must be one of ('identity', 'log', 'categorical'). Found {genex_transform}.")
    if genex_tissue:
        df_merged.columns = ['sequence', 'label']
    else:
        df_merged = df_merged.rename(columns={'seq': 'sequence'})
        genex_values = df_merged.iloc[:, 1:].values.tolist()
        df_merged['label'] = genex_values
        df_merged = df_merged[['sequence', 'label']]
    df_merged_train = df_merged.loc[df_labels['train_test'] == 'train']
    df_merged_test = df_merged.loc[df_labels['train_test'] == 'test']

    df_merged_train.to_csv(outfile_train, index=False, sep='\t')
    df_merged_test.to_csv(outfile_test, index=False, sep='\t')


def main():
    seq_files = SEQ_DIR.glob('*')

    print("Emptying existing files")
    for f in [OUTFILE_ALL_SEQS_TRAIN, OUTFILE_ALL_SEQS_TEST,
              OUTFILE_ALL_SEQS_TRAIN_SAMP, OUTFILE_ALL_SEQS_TEST_SAMP,
              OUTFILE_B73_TRAIN_GENEX, OUTFILE_B73_TEST_GENEX]:
        if f.exists():
            f.unlink()

    print("Processing files")
    for seq_file in seq_files:
        # if seq_file.name == 'refseq.csv':
        #     continue
        print(seq_file)
        process_seq_file(seq_file, OUTFILE_ALL_SEQS_TRAIN,
                         OUTFILE_ALL_SEQS_TEST,
                         seq_key='sequence',
                         test_size=SETTINGS['test_size'],
                         max_seq_len=SETTINGS['max_seq_len'],
                         sample_prob=0.10,
                         outfile_train_sample=OUTFILE_ALL_SEQS_TRAIN_SAMP,
                         outfile_test_sample=OUTFILE_ALL_SEQS_TEST_SAMP)
        if 'b73' in seq_file.name or 'maize' in seq_file.name:
            print(seq_file.name)
            if 'b73' in seq_file.name:
                key = 'seq'
            else:
                key = 'sequence'
            process_seq_file(seq_file, OUTFILE_MAIZE_TRAIN, OUTFILE_MAIZE_TEST,
                             seq_key=key,
                             test_size=SETTINGS['test_size'],
                             max_seq_len=SETTINGS['max_seq_len'])

    print("Processing B73 Genex Data")
    process_genex_data(INFILE_B73, INFILE_B73_GENEX, B73_TRAIN_TEST_LABELS,
                       OUTFILE_B73_TRAIN_GENEX, OUTFILE_B73_TEST_GENEX,
                       genex_tissue=None,
                       max_seq_len=SETTINGS['max_seq_len'])


if __name__ == "__main__":
    main()
