import re
import pandas as pd
import numpy as np
from florabert import config


def extract_name(txt, prefix):
    pattern = prefix + r"[\w\.\_]+\/"
    return re.search(pattern, txt).group()[len(prefix) - 1 : -1]


def test_ensembel():
    """Test ensembel_link"""
    ensembel_df = pd.read_csv(
        config.data_raw / "Ensembl" / "gz_link" / "ensembel_link.csv"
    )
    print(ensembel_df.shape)

    # Test fa links validity
    ensembel_df["extracted_name_fa"] = ensembel_df["fasta_link"].apply(
        lambda x: extract_name(x, r"fasta\/")
    )
    assert np.all(ensembel_df["extracted_name_fa"] == ensembel_df["name"])
    assert np.all(ensembel_df["fasta_link"].str.contains(".dna.toplevel.fa.gz"))

    # Test gff links validity
    ensembel_df["extracted_name_gff"] = ensembel_df["gff3_link"].apply(
        lambda x: extract_name(x, r"gff3\/")
    )
    assert np.all(ensembel_df["extracted_name_gff"] == ensembel_df["name"])
    assert np.all(ensembel_df["gff3_link"].str.contains(".gff3.gz"))


def test_refseq():
    """Test refseq_link"""
    refseq_df = pd.read_csv(config.data_raw / "Refseq" / "gz_link" / "refseq_link.csv")
    print(refseq_df.shape)

    # Test fa links validity
    refseq_df["extracted_name_fna"] = refseq_df["fna_link"].apply(
        lambda x: extract_name(x, r"plant\/")
    )
    assert np.all(
        refseq_df["extracted_name_fna"].str.upper() == refseq_df["name"].str.upper()
    )
    assert np.all(refseq_df["fna_link"].str.endswith("_genomic.fna.gz"))

    # Test gff links validity
    refseq_df["extracted_name_gff"] = refseq_df["gff_link"].apply(
        lambda x: extract_name(x, r"plant\/")
    )
    assert np.all(
        refseq_df["extracted_name_gff"].str.upper() == refseq_df["name"].str.upper()
    )
    assert np.all(refseq_df["gff_link"].str.endswith("genomic.gff.gz"))
