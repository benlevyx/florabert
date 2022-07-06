"""Preparing cultivar and mazize subpopulation data."""
import re

import pandas as pd

from florabert import config


def prepare_col(col):
    return col.lower().replace(" ", "_")


def get_cultivar(assembly_name):
    return re.findall("^Zm-([a-zA-Z0-9]+)-REFERENCE-NAM-1.0$", assembly_name)[0]


def get_gene_prefix(cross_reference):
    return cross_reference.split("->")[1][:-2]


def main():
    df_nam_meta = pd.read_csv(config.data_raw / "Maize_nam" / "meta" / "nam_lines.csv")
    df_nam_meta.columns = [prepare_col(c) for c in df_nam_meta.columns]
    df_nam_meta["cultivar"] = df_nam_meta["assembly_name"].apply(get_cultivar)
    df_nam_meta["gene_prefix"] = df_nam_meta["cross_reference"].apply(get_gene_prefix)
    df_nam_meta = df_nam_meta[["subpopulation", "cultivar", "gene_prefix"]]

    df_nam_meta.to_csv(config.data_final / "nam_data" / "nam_metadata.csv", index=False)


if __name__ == "__main__":
    main()
