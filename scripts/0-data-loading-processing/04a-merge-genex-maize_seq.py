"""[DEPRECATED] Use 04-process-genex-nam.py (will be removed)

Pipeline task for merging the gene sequence and expression data

INPUTS:
    - Inari gene expression data (data/processed/gene_expression/**)
    - MaizeGDB genome sequence data (data/processed/Maize/**)

OUTPUTS:
    - CSV with columns for gene expression data, cultivar metadata,
      foreign key to gene sequence
    - CSV for gene sequence with key to link to gene expression data
"""
import re
import sys
import typing

import pandas as pd
from Bio import SeqIO, Seq
from tqdm import tqdm

from inari import config, utils, dataio


GENEX_DIR = config.data_processed / 'gene_expression'
SEQ_DIR = config.data_processed / 'Maize'
OUTFILE_GENEX = config.data_final / 'merged_genex_data.csv'
OUTFILE_SEQ = config.data_final / 'merged_seq_data.csv'
META_COLS = [
    'Run',
    'growth_condition',
    'Cultivar',
    'Developmental_stage',
    'organism_part',
    'Age'
]


def load_genex_data(sample=None) -> pd.DataFrame:
    df = pd.read_csv(GENEX_DIR / 'TPM_expression_counts_from_25_maize_lines_from_NAM_population.txt', sep='\t', nrows=sample)
    df_meta = df[META_COLS]
    df_genex = df.filter(regex=r'Zm\d{5}[a-z]\d{6}', axis=1)
    df_genex.columns = [c[-6:] for c in df_genex.columns]
    return pd.concat((df_meta, df_genex), axis=1)

def get_cultivar_list() -> list:
    # return [d.name for d in SEQ_DIR.iterdir()]
    # For now, we only process the b73 reference cultivar
    return ['Zmb73']


def get_cultivar_sequences(cultivar: str) -> list:
    filepath = list((SEQ_DIR / cultivar).glob('*.fa'))[0]
    with filepath.open('r') as f:
        sequences = list(SeqIO.parse(f, 'fasta'))
    return sequences


def match_genex_rows_single(df_cultivar: pd.DataFrame, seq: Seq.Seq) -> pd.DataFrame:
    id_ = utils.get_gene_id_num(seq.name)
    
    if not id_ or id_ not in df_cultivar.columns:
        return None
    genex_col = df_cultivar[id_]
    original_col_name = genex_col.name

    chunk = df_cultivar[META_COLS].copy()
    chunk['gene_expression_level'] = genex_col

    # chunk['promoter_sequence'] = str(seq.seq)
    chunk['promoter_name'] = seq.name
    chunk['original_gene_expression_name'] = original_col_name
    return chunk


def match_genex_rows(df_cultivar: pd.DataFrame, 
                     cultivar_sequences: list) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    chunks = []
    names = []
    seqs = []
    cultivar = df_cultivar['Cultivar'].values[0]
    for i, seq in enumerate(tqdm(cultivar_sequences)):
        chunk = match_genex_rows_single(df_cultivar, seq)
        if chunk is None:
            continue
        chunks.append(chunk)
        names.append(seq.name)
        seqs.append(str(seq.seq))
    df_genex = pd.concat(chunks, axis=0)
    df_seq = pd.DataFrame({'name': names, 'seq': seqs})
    return df_genex, df_seq


def get_cultivar_subset(df_genex: pd.DataFrame, cultivar: str) -> pd.DataFrame:
    cultivar_code = re.sub('^Zm', '', cultivar).lower()
    df_cultivar = df_genex[df_genex['Cultivar'].apply(lambda s: s.lower()) == cultivar_code]
    return df_cultivar


def main():
    print("Loading data")
    genex_data = load_genex_data(sample=50)
    cultivar_list = get_cultivar_list()

    for i, cultivar in enumerate(cultivar_list):
        print(f"Cultivar: {cultivar}")
        cultivar_sequences = get_cultivar_sequences(cultivar)
        cultivar_subset = get_cultivar_subset(genex_data, cultivar)
        if cultivar_subset.shape[0] == 0:
            continue
        df_genex, df_seq = match_genex_rows(cultivar_subset, cultivar_sequences)
        for df, outfile in zip((df_genex, df_seq), (OUTFILE_GENEX, OUTFILE_SEQ)):
            df.to_csv(outfile, header=(i == 0), mode=('w' if i == 0 else 'a'))

    print("Done processing")


if __name__ == "__main__":
    main()
