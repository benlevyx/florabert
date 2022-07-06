from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import click

from florabert import config, utils, dataio, metrics
from florabert import transformers as tr
from florabert import visualization as vis


MODEL_PATH = config.models / "transformer" / "full-dataset" / "checkpoint-9500"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
DATA_DIR = config.data_final / "transformer" / "genex" / "nam"
TRAIN_DATA = DATA_DIR / "train.tsv"
TEST_DATA = DATA_DIR / "test.tsv"
LOGDIR = config.data_final / "transformer" / "embeddings"
SAVE_PATH = config.data_final / "transformer" / "kmer"

SAVE_PATH.mkdir(exist_ok=True, parents=True)


@click.command()
@click.option("--nshards", default=None, type=int)
@click.option("--max-kmer", default=6, type=int)
def main(nshards, max_kmer):
    settings = utils.get_model_settings(config.settings, model_name="roberta-lm")
    config_obj, tokenizer, model = tr.load_model(
        "roberta-pred-mean-pool",
        TOKENIZER_DIR,
        pretrained_model=MODEL_PATH,
        **settings,
    )

    datasets = dataio.load_datasets(
        tokenizer,
        DATA_DIR / TEST_DATA,
        seq_key="sequence",
        file_type="csv",
        delimiter="\t",
        log_offset=0.001,
        transformation="log",
        nshards=nshards,
    )

    dataset_test = datasets["train"]

    pred_outputs = metrics.get_predictions(model, dataset_test, return_labels=False)
    del datasets

    torch.save(pred_outputs, SAVE_PATH / "model_true_outputs.pt")

    del pred_outputs

    for k in range(1, max_kmer + 1):
        print(f"kmer flip experiment ({k})")
        datasets_kmer = dataio.load_datasets(
            tokenizer,
            DATA_DIR / TEST_DATA,
            seq_key="sequence",
            file_type="csv",
            delimiter="\t",
            log_offset=0.001,
            transformation="log",
            nshards=nshards,
            kmer=k,
        )
        dataset_test_k = datasets_kmer["train"]
        true_labels, outputs = metrics.get_predictions(model, dataset_test_k)
        idxs = dataset_test_k["index"]
        pd.DataFrame(idxs, columns=["index"]).to_csv(
            SAVE_PATH / f"kmer_{k}_indices.txt", index=False
        )

        del datasets_kmer
        del true_labels

        torch.save(outputs, SAVE_PATH / f"kmer_{k}_output.pt")

        del outputs


if __name__ == "__main__":
    main()
