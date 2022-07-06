"""
Pipeline task for processing gene expression data for NAM lines.

INPUTS:
    - Normalized_TPMs_each_NAM_line_aligned_against_own_refgen/*.txt (gene expression counts for each NAM line)
    - Maize_nam/zm-***.fa (promoter sequences for each NAM line)
    - gene_data_folds.pkl (train/eval/test splits for each gene)

OUTPUTS:
    - csv with promoter sequences, gene expression values, train/eval/test split, and metadata for all cultivars
    - train.tsv, eval.tsv, test.tsv for transformer modelling
"""
import re
import csv
import pickle
from pathlib import PosixPath

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
import numpy as np
from Bio import SeqIO

from florabert import config, utils, dataio


tqdm.pandas()
pandarallel.initialize(progress_bar=True, verbose=1)


GENEX_DIR = (
    config.data_raw
    / "gene_expression"
    / "Normalized_TPMs_each_NAM_line_aligned_against_own_refgen"
)
SEQ_DIR = config.data_processed / "Maize_nam"
B73_DIR = config.data_processed / "Maize" / "Zmb73"
FOLDS_FILE = config.data_final / "nam_data" / "gene_data_folds.pkl"
OUTFILE_ALL_FOLDS = config.data_final / "nam_data" / "merged_seq_genex.csv"
OUTDIR_TRANSFORMER = config.data_final / "transformer" / "genex" / "nam"


def load_cultivar_seqs(cultivar: str):
    files = list((SEQ_DIR / f"zm-{cultivar.lower()}").glob("*.fa"))
    if len(files) >= 1:
        fname = files[0]
        with fname.open("r") as f:
            # Convert to list and remove empty seqs
            seqs = [seq for seq in SeqIO.parse(f, "fasta") if seq.seq]
        return seqs
    return None


def load_b73_seqs():
    files = list(B73_DIR.glob("*.fa"))
    if len(files) >= 1:
        fname = files[0]
        with fname.open("r") as f:
            # Convert to list and remove empty seqs
            seqs = [seq for seq in SeqIO.parse(f, "fasta") if seq.seq]
        return seqs
    return None


def avg_genex_values(df: pd.DataFrame):
    genex_colnames = [c for c in df.columns if c.startswith("Zm")]
    meta_colnames = ["Cultivar", "organism_part"]

    genex_cols = df[genex_colnames]
    meta_cols = df[meta_colnames]
    if meta_cols.shape[0] > 1:
        arr = meta_cols.to_numpy()
        assert (arr[0] == arr).all(axis=0).all(), "metadata columns are not all equal"

    # Average gene expression values row-wise
    genex_values = genex_cols.mean(axis=0)
    return pd.concat((meta_cols.iloc[0], genex_values)).to_frame().T


def filter_genex_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove genes where all TPM counts are less than 1"""
    # return df[df.parallel_apply(lambda row: any([row[tiss] >= 1 for tiss in config.tissues]), axis=1)]
    # def keep_row(row: pd.Series) -> bool:
    #     for x in row:
    #         if x >= 1:
    #             return True
    #     return False
    # return df.copy()[df.apply(keep_row, axis=1)]
    pseudogenes = find_pseudogenes(df)
    return df.loc[~pseudogenes, :].copy()  # .copy() avoids chained assignment


def find_pseudogenes(df: pd.DataFrame) -> pd.Series:
    """Find rows that correspond to pseudogenes (all genes are < 1 TPM).

    Assumes that the dataframe only contains columns corresponding to
    expression values.
    """
    return ~df.progress_apply(lambda row: (row >= 1).any(), axis=1)


def load_genex(fname: PosixPath) -> pd.DataFrame:
    df_genex = pd.read_csv(fname, sep="\t")
    df_genex = df_genex.groupby("organism_part").parallel_apply(avg_genex_values)

    # Transposing so the genes are rows and the tissues are columns
    df_genex = df_genex[
        [c for c in df_genex.columns if c not in ["Cultivar", "organism_part"]]
    ].T
    df_genex.index.name = "gene_id"
    df_genex.columns = [c[0] for c in df_genex.columns]
    df_genex = df_genex.rename(columns={"shoot system": "shoot"})
    return df_genex


def shorten_gene_id(gene_id_long: str) -> str:
    return gene_id_long.split("::")[0]


def prepare_data_transformer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"seq": "sequence"}).copy()
    genex_values = df.loc[:, config.tissues].values.tolist()
    df["labels"] = genex_values
    df = df[["sequence", "labels"]]
    return df


def match_folds(df: pd.DataFrame, folds: dict) -> pd.DataFrame:
    for fold, gene_ids in folds.items():
        genes_to_set = []
        for gene_id_long in tqdm(gene_ids):
            gene_id = shorten_gene_id(gene_id_long)
            if gene_id in df.index:
                genes_to_set.append(gene_id)
        print(len(genes_to_set))
        df.loc[genes_to_set, "fold"] = fold
    return df


def main():
    genex_files = list(
        GENEX_DIR.glob("*_TPM_expression_counts_aligned_against_own_genome.txt")
    )
    with FOLDS_FILE.open("rb") as f:
        folds = pickle.load(f)

    dfs = []
    unfiltered_row_count = 0
    filtered_row_count = 0
    for genex_file in genex_files:
        cultivar = genex_file.name.split("_")[0]
        print(cultivar)

        print("Loading gene expression data")
        df_genex = load_genex(genex_file)
        total_rows = len(df_genex)

        df_filtered = filter_genex_values(df_genex)
        filtered_rows = len(df_filtered)

        unfiltered_row_count += total_rows
        filtered_row_count += filtered_rows

        # Checking filtering worked
        pseudogenes = find_pseudogenes(df_filtered)
        num_pseudogenes = sum(pseudogenes)
        if num_pseudogenes > 0:
            print(df_filtered[pseudogenes])
        assert (
            num_pseudogenes == 0
        ), f"Filtered dataframe still had {num_pseudogenes} pseudogenes. Aborting."

        print(
            f"Kept {filtered_rows} rows out of {total_rows}. Removed {1 - filtered_rows / total_rows:.2%}."
        )

        print("Loading cultivar sequences")
        if cultivar == "B73":
            seqs = load_b73_seqs()
        else:
            seqs = load_cultivar_seqs(cultivar)
        if seqs is None:
            continue

        # Matching genes to gene expression values
        print("Matching gene expression values to sequences")
        for seq in tqdm(seqs):
            gene_id = shorten_gene_id(seq.id)
            if gene_id in df_filtered.index:
                df_filtered.loc[gene_id, "seq"] = str(seq.seq)
                df_filtered.loc[gene_id, "gene_id_full"] = seq.id

        dfs.append(df_filtered)

    df_all_cultivars = pd.concat(dfs, axis=0)

    print(
        f"In total, kept {filtered_row_count} rows out of {unfiltered_row_count}. Removed {1 - filtered_row_count / unfiltered_row_count:.2%}."
    )

    print("Matching folds")
    df_all_cultivars = match_folds(df_all_cultivars, folds)

    # All data
    print("Saving single dataset")
    ordered_cols = ["gene_id_full", "seq", "fold"] + config.tissues
    df_all_cultivars = df_all_cultivars[ordered_cols]

    # Checking for NaN and removing
    nan_rows = df_all_cultivars.apply(lambda x: pd.isna(x).any(), axis=1)
    df_all_cultivars = df_all_cultivars[~nan_rows].copy()
    df_all_cultivars.to_csv(OUTFILE_ALL_FOLDS, index=True)

    # Transformer splits
    print("Splitting folds for transformer data")
    transformer_splits = [
        ("train", ["train_0", "train_1", "train_2", "train_3"]),
        ("eval", ["train_4"]),
        ("test", ["test"]),
    ]
    for name, split in transformer_splits:
        df_split = df_all_cultivars.loc[
            df_all_cultivars["fold"].isin(split), ["seq"] + config.tissues
        ]
        df_split = prepare_data_transformer(df_split)
        df_split.to_csv(OUTDIR_TRANSFORMER / f"{name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
