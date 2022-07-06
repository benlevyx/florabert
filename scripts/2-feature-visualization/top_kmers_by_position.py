"""
Running experiment to determine which kmers are the most important in each
region of the promoter

Regions:
    -1 to -20
    -21 to -100
    -100 to -1000
    
Using 5-mers
"""
from pathlib import PosixPath
from typing import Tuple

import click
import torch
import pandas as pd

from florabert import config, utils, transformers as tr, dataio, metrics


MODEL_PATH = config.models / "transformer" / "b73-included" / "checkpoint-9000"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
DATA_DIR = config.data_final / "transformer" / "genex" / "nam"
TEST_DATA = DATA_DIR / "test.tsv"
SAVE_PATH = config.data_final / "transformer" / "kmer_positions"
SAVE_PATH.mkdir(exist_ok=True, parents=True)


@click.command()
@click.option("--k", default=3, type=int)
@click.option("--pretrained-model", default=MODEL_PATH, type=PosixPath)
@click.option("--output-dir", default=SAVE_PATH, type=PosixPath)
@click.option("--nshards", default=None, type=int)
@click.option("--position-buckets", default="10,100,1000", type=str)
@click.option("--n-repeats", default=1, type=int)
def main(
    k: int,
    pretrained_model: PosixPath,
    output_dir: PosixPath,
    nshards: int,
    position_buckets: Tuple[int],
    n_repeats: int,
):
    settings = utils.get_model_settings(config.settings, model_name="roberta-lm")
    config_obj, tokenizer, model = tr.load_model(
        "roberta-pred-mean-pool",
        TOKENIZER_DIR,
        pretrained_model=pretrained_model,
        **settings,
    )

    datasets = dataio.load_datasets(
        tokenizer,
        TEST_DATA,
        seq_key="sequence",
        file_type="csv",
        delimiter="\t",
        log_offset=0.001,
        transformation="log",
        nshards=nshards,
        shuffle=False,
        min_seq_len=k,
    )

    dataset_test = datasets["train"]

    print("Getting predictions for unmodified dataset")
    pred_outputs = metrics.get_predictions(model, dataset_test, return_labels=False)
    del datasets

    torch.save(pred_outputs, output_dir / "model_true_outputs.pt")

    del pred_outputs

    position_buckets = [int(b) for b in position_buckets.split(",")]
    buckets = []
    for i, b in enumerate(position_buckets):
        if i == 0:
            buckets.append((0, b))
        else:
            buckets.append((position_buckets[i - 1], b))

    for i in range(n_repeats):
        print(f"Repeating experiment: {i + 1}/{n_repeats}")
        datasets_kmer = dataio.load_datasets(
            tokenizer,
            TEST_DATA,
            seq_key="sequence",
            file_type="csv",
            delimiter="\t",
            log_offset=0.001,
            transformation="log",
            nshards=nshards,
            kmer=k,
            position_buckets=buckets,
            shuffle=False,
            min_seq_len=k,
            random_seed=i,
        )
        dataset_test_k = datasets_kmer["train"]

        print("Getting predictions for bucketed and flipped dataset")
        true_labels, outputs = metrics.get_predictions(model, dataset_test_k)
        res = pd.DataFrame(
            {key: dataset_test_k[key] for key in ["index", "kmer_from", "kmer_to"]}
        )
        res.to_csv(output_dir / f"kmer_{k}_indices_{i}.csv", index=False)

        del datasets_kmer
        del true_labels

        torch.save(outputs, output_dir / f"kmer_{k}_output_{i}.pt")

    del outputs


if __name__ == "__main__":
    main()
