import shutil

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from florabert import config, utils, dataio
from florabert import transformers as tr


# Hack to avoid embedding writer error
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


SETTINGS = config.settings["transformer"]
MODEL_PATH = config.models / "transformer" / "prediction-model"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
DATA_DIR = config.data_final / "transformer" / "genex"
TRAIN_DATA = DATA_DIR / "train.tsv"
TEST_DATA = DATA_DIR / "dev.tsv"
LOGDIR = config.data_final / "transformer" / "embeddings"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    return tr.load_model(
        SETTINGS["prediction-model"]["name"],
        TOKENIZER_DIR,
        max_len=SETTINGS["data"]["max_tokenized_len"],
        pretrained_model=MODEL_PATH,
        **SETTINGS["language-model"]["config"],
        **SETTINGS["prediction-model"]["config"]
    )


def load_data(tokenizer):
    return dataio.load_datasets(
        tokenizer,
        TRAIN_DATA,
        TEST_DATA,
        file_type="csv",
        delimiter="\t",
        seq_key="sequence",
    )


def make_summary_writer():
    return torch.utils.tensorboard.SummaryWriter(log_dir=LOGDIR)


def embed_sequences(inputs: dict, model: torch.nn.Module):
    """Generate sequence embeddings for tokenized sequences in `inputs`"""
    model.eval().to(device)
    inputs = {k: inputs[k].to(device) for k in ["attention_mask", "input_ids"]}
    with torch.no_grad():
        embeds = model.embed(**inputs)
    return embeds


def main():
    config_obj, tokenizer, model = load_model()
    args = utils.get_args()
    datasets = load_data(tokenizer)
    tissue_embeddings = model.get_tissue_embeddings()
    loader = DataLoader(
        datasets["test"], batch_size=64, collate_fn=dataio.load_data_collator("pred")
    )

    print("Computing embeddings")
    embeddings = []
    for batch in tqdm(loader):
        embeddings.append(embed_sequences(batch, model))
    embeddings = torch.cat(embeddings, dim=0)

    metadata = [item["sequence"] for item in datasets["test"]]
    if args.num_embeddings > 0:
        idxs = np.random.choice(embeddings.size(0), size=args.num_embeddings)
        mat = torch.cat((tissue_embeddings.cpu(), embeddings[idxs].cpu()), dim=0)
        meta = config.tissues + [metadata[i] for i in idxs]
    else:
        mat = torch.cat((tissue_embeddings.cpu(), embeddings), dim=0)
        meta = config.tissues + metadata

    if LOGDIR.exists():
        shutil.rmtree(LOGDIR)
    writer = make_summary_writer()
    writer.add_embedding(mat, metadata=meta)


if __name__ == "__main__":
    main()
