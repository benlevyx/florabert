"""Short script for getting langauge model model embeddings for various species in the test dataset."""

import re
import shutil
from pathlib import Path, PosixPath
import logging

import click
from florabert.nlp import get_kmer_counts
import numpy as np
import pandas as pd
import dask.dataframe as dd
import tensorboard as tb

# Hack to avoid embedding writer error
import tensorflow as tf
import torch
from florabert import config, utils
from florabert import transformers as tr
from tqdm import tqdm

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = config.data_final / "transformer" / "seq"
INPUT_FILE = config.data_processed / "combined" / "ensembl.csv"
OUTPUT_DIR = config.data_final / "model_output" / "transformer"
MODEL_PATH = config.models / "transformer" / "language-model-finetuned"
TB_DIR = OUTPUT_DIR / "tensorboard"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"


def _load_data(input_file: PosixPath, sample: float) -> pd.DataFrame:
    data_spec = pd.read_csv(
        input_file,
        header=0,
        skiprows=lambda i: i > 0 and np.random.rand() > sample,
    )
    return data_spec


def sample_dataset(dataset, n=10):
    idxs = np.random.choice(len(dataset), size=n, replace=False)
    return dataset[idxs]


def prepare_inputs(batch):
    inputs = torch.tensor(batch["input_ids"]).to(device)
    attn_mask = torch.tensor(batch["attention_mask"]).to(device)
    if len(inputs.size()) == 1:
        inputs = inputs.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)
    return {"input_ids": inputs, "attention_mask": attn_mask}


def get_embeddings(model_outputs, method="avg", layer=-2):
    hidden_states = model_outputs[1]
    layer_hiddens = hidden_states[layer]

    batch_size = layer_hiddens.size(0)
    if method == "avg":
        return torch.mean(layer_hiddens, dim=1)
    elif method == "concat":
        return layer_hiddens.view(batch_size, -1)


def get_all_embeddings(
    data_spec: pd.DataFrame, model, tokenizer, batch_size: int = 64
) -> torch.Tensor:
    species = []
    embeddings = []
    for i in tqdm(range(int(np.ceil(len(data_spec) / batch_size)))):
        rmin = i * batch_size
        rmax = min((i + 1) * batch_size, len(data_spec) + 1)
        rows = data_spec.iloc[rmin:rmax]
        try:
            inputs = tokenizer.batch_encode_plus(
                rows["sequence"].tolist(),
                padding="max_length",
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            species.append(rows["species"].tolist())
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embeds = get_embeddings(outputs)
            embeddings.append(embeds)
        except TypeError:
            break

    # embeddings = torch.cat(embeddings)
    return embeddings, species


@click.command()
@click.option("--load-from", default=None)
@click.option("--sample", default=1.0)
@click.option(
    "--input-file",
    default=INPUT_FILE,
)
@click.option("--output-dir", default=OUTPUT_DIR)
@click.option("--pretrained-model", default=MODEL_PATH)
@click.option("--model-name", default="roberta-lm")
@click.option("--cultivars", default=False, is_flag=True)
@click.option("-k", default=None)
def main(
    load_from,
    sample,
    input_file,
    output_dir,
    pretrained_model,
    model_name,
    cultivars,
    k,
):
    input_file = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()
    pretrained_model = Path(pretrained_model).resolve()

    if not load_from and k is None:
        settings = utils.get_model_settings(config.settings, model_name=model_name)

        log.info("Loading model")
        config_obj, tokenizer, model = tr.load_model(
            model_name,
            TOKENIZER_DIR,
            pretrained_model=pretrained_model,
            **settings,
        )
        model.to(device)

        log.info("Loading data")
        data_spec = _load_data(input_file, sample)

        log.info("Getting embeddings")
        all_embeddings, all_species = get_all_embeddings(
            data_spec, model, tokenizer, batch_size=64
        )

        log.info("Saving embeddings...")
        embed_dir = output_dir / "species_embeddings"
        if embed_dir.exists():
            shutil.rmtree(embed_dir)
        embed_dir.mkdir(parents=True, exist_ok=True)
        max_partition_size = 1e4
        df_embeddings = pd.DataFrame()
        partition_id = 0
        for embed_chunk, spec_chunk in tqdm(list(zip(all_embeddings, all_species))):
            if len(df_embeddings) >= max_partition_size:
                df_embeddings.to_csv(
                    embed_dir / f"partition_{partition_id}.csv", index=False
                )
                df_embeddings = pd.DataFrame()
                partition_id += 1
                continue
            df_chunk = pd.DataFrame(embed_chunk.cpu().numpy())
            df_chunk["species"] = spec_chunk

            df_embeddings = df_embeddings.append(df_chunk)

        del all_embeddings

    elif not load_from and k is not None:
        k = int(k)
        log.info("Using %d-mer embeddings", k)

        log.info("Loading data")
        data_spec = _load_data(input_file, sample)
        bow_rep, kmers = get_kmer_counts(data_spec["sequence"], k)
        df_bow = pd.DataFrame(bow_rep, columns=kmers)
        df_bow["species"] = data_spec["species"]

        log.info("Saving to CSV")
        embed_dir = output_dir / "kmer_embeddings"
        if embed_dir.exists():
            shutil.rmtree(embed_dir)
        embed_dir.mkdir(parents=True, exist_ok=True)
        df_bow.to_csv(embed_dir / f"partition_0.csv", index=False)

    else:
        precomputed_embeddings = load_from

    # Loading from disk as a dask array
    precomputed_embeddings = embed_dir / "partition_*.csv"
    log.info("Loading precomputed embeddings from %s", str(precomputed_embeddings))
    df_embeddings = dd.read_csv(precomputed_embeddings)

    log.info("Generating centroids")
    df_centroids = df_embeddings.groupby("species").mean().compute()

    if cultivars:
        log.info("Getting cultivar types")
        meta = df_centroids.index.to_series().to_frame()
        meta.columns = ["cultivar"]
        cultivar_meta = pd.read_csv(
            config.data_raw / "Maize_nam" / "meta" / "nam_lines.csv"
        )
        cultivar_meta["cultivar"] = cultivar_meta["Assembly name"].apply(
            lambda x: re.findall(r"^(Zm\-[a-zA-Z0-9]+)\-", x)[0].lower()
        )
        cultivar_meta = cultivar_meta.rename(columns={"Subpopulation": "subpopulation"})
        cultivar_meta = cultivar_meta[["cultivar", "subpopulation"]]
        meta = meta.merge(cultivar_meta, on="cultivar", how="left")
    else:

        log.info("Getting taxonomical information from ebi.ac.uk API.")
        species_type = df_centroids.index.to_series().apply(
            utils.get_species_type
        )  # type: pd.Series

        unknown_species_type = {
            "dioscorea_rotundata": "monocot",
            "oryza_indica": "monocot",
            "ostreococcus_lucimarinus": "non-plant",
            "physcomitrella_patens": "non-embryophyte",
        }
        species_type.loc[species_type == "unknown"] = (
            species_type.loc[species_type == "unknown"]
            .index.to_series()
            .map(unknown_species_type)
        )

        # log.info("Saving tensorboard embeddings")
        # if TB_DIR.exists():
        #     shutil.rmtree(TB_DIR)

        meta = species_type.to_frame().rename(columns={"species": "type"}).reset_index()
        meta = meta[["species", "type"]]
    # writer = SummaryWriter(log_dir=TB_DIR)
    # writer.add_embedding(
    #     df_centroids.values,
    #     metadata=meta.to_records(index=False).tolist(),
    #     metadata_header=["species", "type"],
    # )

    log.info("Saving metadata")
    df_centroids.to_csv(output_dir / "species_centroids.csv", index=True)
    meta.to_csv(output_dir / "species_metadata.csv", index=False)

    log.info("Done")


if __name__ == "__main__":
    main()
