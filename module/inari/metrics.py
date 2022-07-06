"""Reusable metrics functions for evaluating models
"""
import multiprocessing as mp
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm

from .utils import compute_pseudo_r2, compute_r2


def get_predictions(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    return_labels: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute model predictions for `dataset`.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataset (torch.utils.data.Dataset): Dataset to get predictions for
        return_labels (bool, optional): Whether to return the labels (predictions are always returned).
            Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 'true_labels', 'pred_labels'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=default_data_collator,
        drop_last=False,
        num_workers=mp.cpu_count(),
    )

    true_labels = []
    pred_labels = []
    for batch in tqdm(loader):
        inputs = {k: batch[k].to(device) for k in ["attention_mask", "input_ids"]}
        with torch.no_grad():
            outputs = model(**inputs)
        del inputs  # to free up space on GPU
        logits = outputs[0]

        if return_labels:
            true_labels.append(batch["labels"].cpu())
        pred_labels.append(logits.cpu())

    print("Concatenating")
    pred_labels = torch.cat(pred_labels, dim=0)  # pred_labels is list of Tensor
    if not return_labels:
        return pred_labels

    true_labels = torch.cat(true_labels, dim=0)  # true_labels is list of list

    return true_labels, pred_labels


def evaluate_model(
    trg_true: torch.Tensor, trg_pred: torch.Tensor, metric_fns: list
) -> list:
    """Compute model performance for each metric in `metric_fns`

    Args:
        trg_true (torch.Tensor): true labels (n x num_outputs)
        trg_pred (torch.Tensor): predicted labels (n x num_outputs)
        metric_fns (list): metric functions

    Returns:
        list: model performance on each metric, in the same order as `metric_fns`
    """
    return [metric_fn(trg_pred, trg_true) for metric_fn in metric_fns]


def make_tissue_loss(tissue_idx, metric="mse") -> callable:
    """Make a tissue-specific loss function

    Args:
        tissue_idx ([type]): index of the tissue
        metric (str, optional): metric to evaluate. Defaults to 'mse'.

    Returns:
        callable: tissue-specific loss function
    """

    def loss(logits, labels):
        if metric == "mse":
            loss_fn = torch.nn.MSELoss()
        elif metric == "mae":
            loss_fn = make_mae_loss()
        elif metric == "rmse":
            mse_loss = torch.nn.MSELoss()

            def loss_fn(x, y):
                return torch.sqrt(mse_loss(x, y))

        elif metric == "r2":
            loss_fn = compute_r2
        elif metric == "pseudo-r2":
            loss_fn = compute_pseudo_r2

        return loss_fn(logits[:, tissue_idx], labels[:, tissue_idx])

    return loss


def make_mae_loss() -> callable:
    """Make a function computing MAE (L1) loss

    Returns:
        callable: MAE loss
    """
    mae_loss = torch.nn.L1Loss()

    def loss_fn(x, y):
        return mae_loss(torch.exp(x) - 1, torch.exp(y) - 1)

    return loss_fn
