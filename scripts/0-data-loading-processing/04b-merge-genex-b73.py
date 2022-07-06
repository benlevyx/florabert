"""[DEPRECATED] Use 04-process-genex-nam.py (will be removed)

[Temporary] pipeline task for merging sequence and expression data for B73
cultviar only.

INPUTS:
    - df_STAR_HTSeq_v3_genes_counts_B73only_match_based_on_genet_dist_DESeq2_normed_rounded.txt
    - gene_model_xref_v3.txt
    - Zmb73 MaizeGDB sequences

OUTPUTS:
    - CSV with columns for gene expression data, sample metadata, foreign key
      to gene sequence
    - CSV for gene sequences with key to link to gene expression data
"""
import re
import typing

import pandas as pd
from tqdm import tqdm
from Bio import SeqIO, Seq

from inari import config, utils, dataio

GENEX_DIR = config.data_raw / 'gene_expression'
GENEX_FILE = GENEX_DIR / 'df_STAR_HTSeq_v3_genes_counts_B73only_match_based_on_genet_dist_DESeq2_normed_rounded.txt'
XREF_FILE = config.data_processed / 'gene_expression' / 'gene_model_xref_v3.txt'
SEQ_DIR = config.data_processed / 'Ensembl' / 'zea_mays'

CULTIVAR = 'Zmb73'
OUTDIR = config.data_final / CULTIVAR
if not OUTDIR.exists():
    OUTDIR.mkdir()

OUTFILE_GENEX = OUTDIR / 'merged_genex_data_b73.csv'
OUTFILE_SEQ = OUTDIR / 'merged_seq_data_b73.csv'

META_COLS = [
    'tissue',
    'X.Trait.'
]
GENE_REGEX = re.compile(r'(Zm\d{5}[a-z]\d{6})::')


def load_genex_data(sample=None) -> pd.DataFrame:
    df = pd.read_csv(GENEX_FILE, sep='\t', nrows=sample)
    regex_tissue = re.compile(r'(LMAN|LMAD|LMid|L3Base|L3Tip|GShoot|GRoot|Kern)')
    df['tissue'] = df['X.Trait.'].apply(lambda x: re.findall(regex_tissue, x)[0])
    return df


def load_xref_dict() -> pd.DataFrame:
    df = pd.read_csv(XREF_FILE, sep='\t')
    xref_dict = {}
    ref_col = df['v3_gene_model']
    b73_col = df['B73(Zm00001d.2)']
    
    assert len(ref_col) == len(b73_col), 'Length of columns are not equal'

    idx_not_na = ~pd.isna(b73_col)
    for k, v in zip(b73_col[idx_not_na], ref_col[idx_not_na]):
        if ~pd.isna(v):
            if ',' in k:
                for ki in k.split(','):
                    xref_dict[ki] = v
            else:
                xref_dict[k] = v

    return xref_dict


def load_sequences() -> list:
    seq_file = list((SEQ_DIR).glob('*.fa'))[0]
    with seq_file.open('r') as f:
        sequences = list(SeqIO.parse(f, 'fasta'))
    return sequences


def match_genex_single(genex_data: pd.DataFrame,
                       seq: Seq.Seq,
                       xref_dict: dict) -> pd.DataFrame:
    gene_id = utils.get_gene_id_num(seq.name, regex=GENE_REGEX)
    if not gene_id:
        return None
    
    xref_id = xref_dict.get(gene_id, None)
    if not xref_id:
        return None
    
    if xref_id not in genex_data.columns:
        return None
    genex_col = genex_data[xref_id]
    chunk = genex_data[META_COLS].copy()
    chunk['gene_expression_level'] = genex_col
    chunk['promoter_name'] = seq.name
    chunk['v3_gene_model'] = xref_id
    return chunk


def match_genes(genex_data: pd.DataFrame,
                gene_sequences: list,
                xref_dict: dict
                ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    chunks = []
    names = []
    seqs = []
    for i, seq in enumerate(tqdm(gene_sequences)):
        chunk = match_genex_single(genex_data, seq, xref_dict)
        if chunk is None:
            continue
        chunks.append(chunk)
        names.append(seq.name)
        seqs.append(str(seq.seq))
    df_genex = pd.concat(chunks, axis=0)
    df_seq = pd.DataFrame({'name': names, 'seq': seqs})
    return df_genex, df_seq


def main():
    print("Loading data")
    genex_data = load_genex_data()
    
    print("Loading B73 sequences")
    gene_sequences = load_sequences()

    print("Loading XREF dictionary")
    xref_dict = load_xref_dict()

    df_genex, df_seq = match_genes(genex_data, gene_sequences, xref_dict)
    for df, outfile in zip((df_genex, df_seq), (OUTFILE_GENEX, OUTFILE_SEQ)):
        df.to_csv(outfile)


if __name__ == '__main__':
    main()
