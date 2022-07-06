"""Utilities for making visualizations of model performance and data.
"""
import numpy as np

import multiprocessing as mp
from pathlib import PosixPath
from typing import Union, List
import random
import itertools

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from transformers import default_data_collator
from datasets import load_dataset

from . import transformers as tr
from .dataio import make_preprocess_function
from .utils import convert_str_to_tnsr
from .config import settings


# function to flip a random k-mer in the sequence. The output seq size is the same as the original seq size
def make_kmer_flip_function(seq_key, kmer):
    """return a function that will perform random k-mer flip
    Args:
        seq_key (str, optional): Column name of sequence data Can be 'sequence', 'seq', or 'text'. Defaults to 'sequence'.
        kmer(int): numebr k for the k-mer

    Returns:
        callable: k-mer flip function
    """

    def kmer_flip(examples: list):
        # Do the kmer flipping with a batch of exmaples
        if seq_key:
            seqs = examples[seq_key]
        else:
            seqs = examples
        flipped_index = []
        bases = ["A", "T", "G", "C"]
        k_mer_list = ["".join(p) for p in itertools.product(bases, repeat=kmer)]
        temp_seqs = seqs.copy()
        for i in range(len(seqs)):
            list1 = list(seqs[i])
            # Choose an index to flip
            flip_num = random.randint(0, 1000 - kmer + 1)
            flipped_index.append([flip_num])
            list1[flip_num : flip_num + kmer] = np.random.choice(k_mer_list, 1)[0]
            temp_seqs[i] = "".join(list1)
        flipped_sequence = temp_seqs

        # Return a dict
        return {"sequence": flipped_sequence, "index": flipped_index}

    # What you return will be added to the dataset, overriding existing fields that share the same keys
    return kmer_flip


def load_datasets_kmer(
    tokenizer,
    train_data: Union[str, PosixPath],
    test_data: Union[str, PosixPath] = None,
    file_type: str = "csv",  # Use csv
    delimiter: str = "\t",  # Use \t
    seq_key: str = "sequence",  # Use sequence
    kmer: int = 2,
    shuffle: bool = True,
    **kwargs,
):
    """Load and cache data using Huggingface datasets library, with the additional k-mer flip preprocessing
    Args:
        tokenizer (PreTrainedTokenizer): tokenizer to apply to the sequences
        train_data (Union[str, PosixPath]): location of training data
        test_data (Union[str, PosixPath], optional): location of test/evaluation data. Defaults to None.
        file_type (str, optional): type of file. Possible values are 'text' and 'csv'. Defaults to 'csv'.
        delimiter (str, optional): Defaults to '\t'.
        seq_key (str, optional): Column name of sequence data Can be 'sequence', 'seq', or 'text'. Defaults to 'sequence'.
        kmer(int): numebr k for the k-mer
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
    Returns:
        Dataset: [description]
    """
    data_files = {"train": str(train_data)}
    if test_data:
        data_files["test"] = str(test_data)
    if file_type == "csv":
        kwargs.update({"delimiter": delimiter})
    # load_dataset is the HuggingFace function that uses Apache Arrow
    datasets = load_dataset(file_type, data_files=data_files, **kwargs)
    # Probably define this elsewhere, but this is the function that flips the kmers
    kmer_flip = make_kmer_flip_function(seq_key=seq_key, kmer=kmer)
    preprocess_fn = make_preprocess_function(tokenizer, seq_key=seq_key)
    seed = settings["random_seed"]  # settings is config.settings
    dataset = (
        datasets.map(kmer_flip, batched=True)  # Add the mapping here
        .map(preprocess_fn, batched=True)
        .map(convert_str_to_tnsr, batched=True)
    )
    if shuffle:
        dataset = dataset.shuffle(seeds={"train": seed, "test": seed})
    return dataset


def model_pred(model, tokenizer, dataset_train, device="cpu"):
    """Perform gene expression prediciton and return output and rmse
    Args:
        model_name (str): Name of model. Acceptable options are
                - 'roberta-lm',
                - 'roberta-pred',
                - 'roberta-pred-mean-pool'
        tokenizer (PreTrainedTokenizer): tokenizer to apply to the sequences
        dataset_train [description]: training data/testing data
        device (str): 'cpu' or 'gpu'
    Returns:
        tuple: prediciton rmse, prediction output
    """
    true_rmse = []
    true_outputs = []
    model.eval()
    loader = DataLoader(
        dataset_train,
        batch_size=64,
        collate_fn=default_data_collator,
        drop_last=False,
        num_workers=mp.cpu_count(),
    )
    for batch in tqdm(loader):
        inputs = {k: batch[k].to(device) for k in ["attention_mask", "input_ids"]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs[0].to(device)
        loss_fn = torch.nn.MSELoss(reduction="none").to(device)
        curr_mse = (
            torch.sum(loss_fn(logits, batch["labels"].to(device)), dim=1, keepdim=True)
            / 10
        ).to(device)
        curr_rmse = torch.sqrt(curr_mse)
        true_rmse.append(curr_rmse)
        true_outputs.append(logits)

    return true_rmse, true_outputs


def flatten(output):  # flatten the output so that we can save the results
    """Flatten the torch tensor output into a list of np array
    Args:
        output (torch.tensor): model prediction output
    Returns:
        list: list contains flattened output
    """
    py_ls = [i.squeeze().tolist() for i in tqdm(output)]
    flat_ls = [item for sublist in tqdm(py_ls) for item in sublist]
    return flat_ls


def ridgeplot(df, row, value, xlim=None):
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(df, row=row, hue=row, aspect=15, height=0.5, palette=pal)

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        value,
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, value, clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        if xlim is not None:
            ax.set_xlim(*xlim)

    g.map(label, value)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)


def scatter_genex_predictions(
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    save_dir: PosixPath,
    tissue_names: List[str] = None,
    **plotting_kwargs,
):
    """Scatter true labels (x) vs. predicted labels (y)

    Args:
        true_labels (torch.Tensor): (num_genes x num_tissues)
        pred_labels (torch.Tensor): (num_genes x num_tissues)
        save_dir (PosixPath): Directory to save figures in
        tissue_names (List[str], optional): Titles for each panel.
            Defaults to number of column
        plotting_kwargs: additional arguments to pass to `plt.scatter`.
    """
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()

    if "color" not in plotting_kwargs:
        plotting_kwargs.update({"color": "blue"})
    if "alpha" not in plotting_kwargs:
        plotting_kwargs.update({"alpha": 0.3})
    if "s" not in plotting_kwargs:
        plotting_kwargs

    num_tissues = true_labels.shape[1]
    if tissue_names is None:
        titles = np.arange(num_tissues)
    else:
        titles = tissue_names

    for i, title in enumerate(titles):
        plt.scatter(true_labels[:, i], pred_labels[:, i], **plotting_kwargs)
        plt.title(title)
        ax = plt.gca()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        plt.xlabel("True expression (log-TPM)")
        plt.ylabel("Predicted expression (log-TPM)")
        plt.savefig(save_dir / f"true_vs_pred_{title}.png", dpi=100)
        plt.close()
