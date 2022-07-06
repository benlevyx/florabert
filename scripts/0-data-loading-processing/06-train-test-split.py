"""
    Script used to divide the promoter sequences into train and test sets, splitting the training
    data into 5 folds that could be used for cross-validation while taking into consideration
    gene clusters based on identity similarity (refer to 05a-cluster-maize_seq.sh).

    Inputs:
        maize_nam.csv_cluster.tsv: clusters of genes associated with reference genes

    Outputs:
        gene_data_folds.pkl: a dictionary indicating train/test split and folds. Note, use
                             `with open(str(save_path), 'rb') as pfile: data_folds = pickle.load(pfile)`
                             to load the dict
"""
import pickle
import pandas as pd
import numpy as np
from florabert import config
from tqdm import tqdm

np.random.seed(47)

# Global Vars
TRAIN_SIZE = config.settings["TRAIN_SIZE"]


def main():
    # Read in the cluster tsv
    print("Reading clustered data")
    cluster_path = (
        config.data_processed / "combined" / "clustered" / "maize_nam.csv_cluster.tsv"
    )
    cluster_df = pd.read_csv(cluster_path, sep="\t", header=None)

    # Compile the clusters into a dict
    print("compiling clusters into a dict")
    ref_dict = {}
    for _, (ref, gene) in tqdm(list(cluster_df.iterrows())):
        if ref not in ref_dict:
            ref_dict[ref] = [ref, gene]
        else:
            ref_dict[ref].append(gene)
    print("Total number of clusters:", len(ref_dict.keys()))

    # Create a random list of reference genes
    ref_rand = list(ref_dict.keys())
    np.random.shuffle(ref_rand)

    # Create the storage for data foles
    folds = 5
    data_folds = {f"train_{i}": [] for i in range(folds)}
    data_folds["test"] = []

    # Create the test set
    test_len = (1 - TRAIN_SIZE) * len(cluster_df[1])
    train_start_id = 0
    for i, ref in enumerate(ref_rand):
        data_folds["test"].extend(ref_dict[ref])
        if len(data_folds["test"]) >= test_len:
            train_start_id = i + 1
            break

    # Create the 5 fold for train sets
    for i, ref in enumerate(ref_rand[train_start_id:]):
        data_folds[f"train_{i % folds}"].extend(ref_dict[ref])
    print("Length of each fold:", [(k, len(v)) for k, v in data_folds.items()])

    # Sanity checks
    all_seq = []
    for v in data_folds.values():
        all_seq.extend(v)
    assert (
        sum([len(v) for v in data_folds.values()])
        == sum([len(v) for v in ref_dict.values()])
        == len(all_seq)
    )

    # Save the folds to dict
    save_path = config.data_final / "nam_data" / "gene_data_folds.pkl"
    with open(str(save_path), "wb") as pfile:
        pickle.dump(data_folds, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(str(save_path), "rb") as pfile:
        reloaded = pickle.load(pfile)
    assert sum([len(v) for v in reloaded.values()]) == len(all_seq)


if __name__ == "__main__":
    main()
