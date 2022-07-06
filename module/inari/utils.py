import os
import sys
import wget
import requests
import re
import argparse
from types import GeneratorType, ModuleType
from typing import Union, Tuple
import subprocess
from pathlib import PosixPath, Path
import importlib as im
import json
import pickle

import pandas as pd
import numpy as np
from IPython.display import display
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score

from .config import settings, output, data_final, models


def download(url, output_path, db_name, dna_name=None):
    """Uses wget to download file at url to output_path.
    Params:
        url: url of file to be downloaded
        output_path: path of the output file
        dna_name: str, name for dna .fa/fna file
        db_name: str, name of the database to be processed
    """
    if db_name == "Ensembl":
        if len(os.listdir(output_path)) == 0:
            output_path = str(output_path)
            wget.download(url, output_path)
    else:
        r = requests.get(url, stream=True)
        with open(str(output_path) + "/" + str(dna_name) + ".gz", "wb") as f:
            for chunk in r.raw.stream(1024, decode_content=False):
                if chunk:
                    f.write(chunk)


def execute(exe_str):
    """Execute the given command using os.system
    Params:
        exe_str: str, the exe string to be executed
    """
    response = os.system(exe_str)
    if "grep" not in exe_str:
        # The response should be 0 in all cases other than grep
        assert response == 0


def unzip(folder_path, file_name):
    """Upzips the .gz file w/ <file_name>.gz to the given <folder_path>
    Params:
        folder_path: str, the folder that contains the gz file
        file_name: str, name of the file to be unziped
    """
    out_path = os.path.join(folder_path, file_name)
    in_path = out_path + ".gz"
    exe_str = f"gunzip  -c {in_path} > {out_path}"
    execute(exe_str)


def display_all(df, columns=True, rows=True, cols=None):
    """Display all columns and/or rows of a dataframe in a jupyter notebook
    or IPython terminal.

    Args:
        df ([type]): [description]
        columns (bool, optional): [description]. Defaults to True.
        rows (bool, optional): [description]. Defaults to True.
    """
    args = []
    if cols is not None:
        col_flag = cols
    else:
        col_flag = columns
    if col_flag:
        args.extend(("display.max_columns", None))
    if rows:
        args.extend(("display.max_rows", None))

    with pd.option_context(*args):
        display(df)


def clear_folder(folder_path, to_continue, exclude=None):
    if not to_continue:
        to_continue = input(
            "DANGER: are you sure you want to delete {file_name} at {folder_path} (y/n)?"
        )
    if to_continue == "y":
        if exclude:
            to_delete = [f for f in os.listdir(folder_path) if exclude not in f]
        else:
            to_delete = os.listdir(folder_path)
        for file_name in to_delete:
            execute("rm " + os.path.join(folder_path, file_name))
    else:
        print("Action aborted.")


def get_gene_id_num(gene_name: str, regex=None) -> str:
    """Convert a gene name in the format 'Zm00020a000002::chr1:21757-22757(-)' to
    the 6-digit gene identifier

    Args:
        gene_name (str): The gene name in the above format
        regex (str): Optional, the pattern to match, with the first group
                     being extracted and returned
    """
    if regex is None:
        regex = r"Zm\d{5}[a-z](\d{6})::"
    match = re.match(regex, gene_name)
    if match:
        return match.group(1)
    else:
        return None


def is_iterable(obj) -> bool:
    """Return True if `obj` is an iterable (not string)"""
    return type(obj) in (list, tuple, GeneratorType)


def ensure_iterable(obj) -> list:
    return obj if is_iterable(obj) else [obj]


def count_lines(f) -> int:
    """Use `wc` to count number of lines in files (only works on linux)"""
    resp = subprocess.check_output(["wc", "-l", str(f)])
    return int(resp.decode("utf-8").split()[0])


def load_script_as_module(script_path: Union[str, PosixPath]) -> ModuleType:
    """Load a script as a python module."""
    if type(script_path) == str:
        script_path = Path(script_path)
    module_dir = str(script_path.parent)
    module_file = re.sub(".py$", "", str(script_path.name))

    if module_dir not in sys.path:
        sys.path.append(module_dir)

    return im.import_module(module_file)


def genex_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        values="gene_expression_level", index="promoter_name", columns="tissue"
    )


def train_val_split(
    df: pd.DataFrame, p_val: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_size = int(np.floor(len(df) * p_val))
    val_idxs = np.random.choice(df.index, size=val_size, replace=False)
    train_data = df.loc[~df.index.isin(val_idxs)]
    val_data = df.loc[val_idxs]
    return train_data, val_data


def save_json(dct: dict, savepath: PosixPath):
    with savepath.open("w") as f:
        json.dump(dct, f)


def combine_vocab_dicts(*args) -> dict:
    """Combine a variable number of dictionaries, each of which will be
    added to the vocab with indices beginning at the sum of the max indices
    of the dicts before it.
    """
    res = {}
    max_idx = 0
    for vocab_dict in args:
        for k, v in vocab_dict.items():
            res.update({k: v + max_idx})
        max_idx = max(vocab_dict.values()) + max_idx
    return res


def get_activation_fn(activation) -> Union[torch.nn.Module, callable]:
    if type(activation) == callable:
        f = activation
    elif activation == "relu":
        f = torch.nn.ReLU()
    elif activation == "linear":

        def f(x):
            return x

    elif activation == "elu":
        f = torch.nn.ELU()
    elif activation == "tanh":
        f = torch.nn.Tanh()
    return f


def preprocess_genex(genex_data: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if settings["data"].get("preprocess", False):
        preproc_dict = settings["data"]["preprocess"]
        preproc_type = preproc_dict["type"]
        if preproc_type == "log":
            delta = preproc_dict["delta"]
            df_preprocessed = genex_data.applymap(lambda x: np.log(x + delta))
        elif preproc_type == "binary":
            thresh = preproc_dict["threshold"]
            df_preprocessed = genex_data.applymap(lambda x: float(x > thresh))
        elif preproc_type == "ceiling":
            ceiling = preproc_dict["ceiling"]
            df_preprocessed = genex_data.applymap(lambda x: min(ceiling, x))
        else:
            df_preprocessed = genex_data
        return df_preprocessed
    else:
        return genex_data


def compute_model_metrics(
    model: torch.nn.Module,
    dataset_train: torch.nn.Module,
    dataset_val: torch.nn.Module = None,
    dataset_test: torch.nn.Module = None,
    batch_size: int = 2048,
) -> dict:
    metrics = {}
    model.cpu()
    model.eval()
    for name, dataset in zip(
        ("train", "val", "test"), (dataset_train, dataset_val, dataset_test)
    ):
        if dataset is not None:
            metrics[name] = compute_model_metrics_single_dataset(
                model, dataset, batch_size=batch_size
            )
    return metrics


def compute_model_metrics_single_dataset(
    model: torch.nn.Module, dataset: torch.nn.Module, batch_size: int = 2048
) -> dict:
    """Compute performance metrics for `model` on `dataset`.

    The following metrics are computed:
        1. MSE
        2. MAE
        3. Classification accuracy (expressed/not expressed)
        4. Pseudo-R2

    Args:
        model (torch.nn.Module): the model to evaluate
        dataset (torch.nn.Module): the dataset to evaluate

    Returns:
        'metrics' (dict): dictionary of computed metrics
    """
    metrics = {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    trg_tnsrs = []
    pred_tnsrs = []
    for inputs, trg in tqdm(loader):
        pred = model(inputs)
        pred_tnsrs.append(pred)
        trg_tnsrs.append(trg)

    pred = torch.cat(pred_tnsrs, dim=0)
    trg = torch.cat(trg_tnsrs, dim=0)

    # 1. MSE
    metrics["mse"] = torch.nn.functional.mse_loss(pred, trg).item()

    # 2. MAE
    metrics["mae"] = torch.nn.functional.l1_loss(pred, trg).item()

    # 4. Pseudo-R2
    ybar = torch.sum(trg).item()
    ss_tot = torch.sum((trg - ybar) ** 2).item()
    ss_res = torch.sum((trg - pred) ** 2).item()
    metrics["r2"] = 1 - ss_res / ss_tot

    # 3. Classification
    expr_thresh = settings["expressed_threshold"]
    expr_true = (trg >= expr_thresh).int()
    expr_pred = (pred >= expr_thresh).int()

    correct_pred_rate = torch.mean((1 - (expr_true - expr_pred) ** 2).float(), dim=0)
    metrics["classification_accuracy"] = json.dumps(
        str(list(correct_pred_rate.numpy()))
    )

    return metrics


def get_args(
    data_dir=data_final / "transformer" / "seq",
    train_data="all_seqs_train.txt",
    eval_data=None,
    test_data="all_seqs_test.txt",
    output_dir=models / "transformer" / "language-model",
    model_name=None,
    pretrained_model=None,
    tokenizer_dir=None,
    log_offset=None,
    preprocessor=None,
    filter_empty=False,
    hyperparam_search_metrics=None,
    hyperparam_search_trials=None,
    transformation=None,
    output_mode=None,
) -> argparse.Namespace:
    """Use Python's ArgumentParser to create a namespace from (optional) user input

    Args:
        data_dir ([type], optional): Base location of data files. Defaults to data_final/'transformer'/'seq'.
        train_data (str, optional): Name of train data file in `data_dir` Defaults to 'all_seqs_train.txt'.
        test_data (str, optional): Name of test data file in `data_dir`. Defaults to 'all_seqs_test.txt'.
        output_dir ([type], optional): Location to save trained model. Defaults to models/'transformer'/'language-model'.
        model_name (Union[str, PosixPath], optional): Name of model
        pretrained_mdoel (Union[str, PosixPath], optional): path to config and weights for huggingface pretrained model.
        tokenizer_dir (Union[str, PosixPath], optional): path to config files for huggingface pretrained tokenizer.
        filter_empty (bool, optional): Whether to filter out empty sequences.
            Necessary for kmer-based models; takes additional time.
        hyperparam_search_metrics (Union[list, str], optional): metrics for hyperparameter search.
        hyperparam_search_trials (int, optional): number of trials to run hyperparameter search.
        transformation (str, optional): how to transform data. Defaults to None.
        output_mode (str, optional): default output mode for model and data transformation. Defaults to None.
    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--warmstart",
        action="store_true",
        help="Whether to start with a saved checkpoint",
        default=False,
    )
    parser.add_argument("--num-embeddings", type=int, default=-1)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(data_dir),
        help="Directory containing train/eval data. Defaults to data/final/transformer/seq",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=train_data,
        help="Name of training data file. Will be added to the end of `--data-dir`.",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=eval_data,
        help="Name of eval data file. Will be added to the end of `--data-dir`.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=test_data,
        help="Name of test data file. Will be added to the end of `--data-dir`.",
    )
    parser.add_argument("--output-dir", type=str, default=str(output_dir))
    parser.add_argument(
        "--model-name",
        type=str,
        help='Name of model. Supported values are "roberta-lm", "roberta-pred", "roberta-pred-mean-pool", "dnabert-lm", "dnabert-pred", "dnabert-pred-mean-pool"',
        default=model_name,
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Directory containing config.json and pytorch_model.bin files for loading pretrained huggingface model",
        default=(str(pretrained_model) if pretrained_model else None),
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        help="Directory containing necessary files to instantiate pretrained tokenizer.",
        default=str(tokenizer_dir),
    )
    parser.add_argument(
        "--log-offset",
        type=float,
        help="Offset to apply to gene expression values before log transform",
        default=log_offset,
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        help="Path to pickled preprocessor file",
        default=preprocessor,
    )
    parser.add_argument(
        "--filter-empty",
        help="Whether to filter out empty sequences.",
        default=filter_empty,
        action="store_true",
    )
    parser.add_argument(
        "--tissue-subset", default=None, help="Subset of tissues to use", nargs="*"
    )
    parser.add_argument("--hyperparameter-search", action="store_true", default=False)
    parser.add_argument("--ntrials", default=hyperparam_search_trials, type=int)
    parser.add_argument("--metrics", default=hyperparam_search_metrics, nargs="*")
    parser.add_argument("--direction", type=str, default="minimize")
    parser.add_argument(
        "--nshards",
        type=int,
        default=None,
        help="Number of shards to divide data into; only the first is kept.",
    )
    parser.add_argument(
        "--nshards-eval",
        type=int,
        default=None,
        help="Number of shards to divide eval data into.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum value for filtering gene expression values.",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default=transformation,
        help='How to transform the data. Options are "log", "boxcox"',
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze the pretrained base of the model",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        help='Output mode for model: {"regression", "classification"}',
        default=output_mode,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for training. Default None",
        default=None,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        help="Number of epochs to train for",
        default=None,
    )
    parser.add_argument(
        "--search-metric",
        type=str,
        help="Metric to optimize in hyperparameter search",
        default=None,
    )
    parser.add_argument("--batch-norm", action="store_true", default=False)
    args = parser.parse_args()

    if args.pretrained_model and not args.pretrained_model.startswith("/"):
        args.pretrained_model = str(Path.cwd() / args.pretrained_model)

    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)

    args.train_data = _get_fpath_if_not_none(args.data_dir, args.train_data)
    args.eval_data = _get_fpath_if_not_none(args.data_dir, args.eval_data)
    args.test_data = _get_fpath_if_not_none(args.data_dir, args.test_data)

    args.preprocessor = Path(args.preprocessor) if args.preprocessor else None

    if args.tissue_subset is not None:
        if isinstance(args.tissue_subset, (int, str)):
            args.tissue_subset = [args.tissue_subset]
        args.tissue_subset = [
            int(t) if t.isnumeric() else t for t in args.tissue_subset
        ]
    return args


def _get_fpath_if_not_none(
    dirpath: PosixPath, fpath: PosixPath
) -> Union[None, PosixPath]:
    if fpath:
        return dirpath / fpath
    return None


def get_latest_checkpoint(directory: PosixPath) -> PosixPath:
    """Return the latest checkpoint in `directory` (with the highest number)."""
    checkpoints = list(directory.glob("checkpoint-*"))
    max_ckpt = -1
    for i, ckpt in enumerate(checkpoints):
        num = int(ckpt.name.split("-")[-1])
        if num > max_ckpt:
            max_ckpt = num
    return directory / f"checkpoint-{max_ckpt}"


def count_model_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count number of model parameters

    Args:
        model (torch.nn.Module): Model with parameters to count
        trainable_only (bool, optional): Whether to count only parameters with `p.requires_grad = True` . Defaults to True.

    Returns:
        int: Number of (trainable) model parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_model_performance(eval_df: pd.DataFrame, model_name: str):
    """Saves model performance to output/ dir"""
    root = output / "model_eval"
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    save_path = root / (model_name + ".csv")
    eval_df.to_csv(save_path, index=False)


def convert_str_to_tnsr(examples: dict) -> dict:
    """Utility function to load vector-valued data

    Args:
        examples (str): Vector data in form of string

    Returns:
        dict
    """
    strs = [li.replace("[", "").split("],") for li in examples["labels"]]
    lists = [list(map(float, s.replace("]", "").split(","))) for li in strs for s in li]
    # tnsrs = [torch.tensor(li) for li in lists]
    return {"labels": lists}


def get_model_settings(
    settings: dict, args: dict = None, model_name: str = None
) -> dict:
    """Get the appropriate model settings from the dictionary `settings`."""
    if model_name is None:
        model_name = args.model_name
    base_model_name = model_name.split("-")[0] + "-base"
    base_model_settings = settings["models"].get(base_model_name, {})
    model_settings = settings["models"].get(model_name, {})
    data_settings = settings["data"]
    settings = dict(**base_model_settings, **model_settings, **data_settings)

    if args is not None:
        if args.output_mode:
            settings["output_mode"] = args.output_mode
        if args.tissue_subset is not None:
            settings["num_labels"] = len(args.tissue_subset)
        if args.batch_norm:
            settings["batch_norm"] = args.batch_norm

    return settings


def _ensure1d(arr: np.ndarray):
    if type(arr) == torch.Tensor:
        arr = arr.cpu().numpy()
    if len(arr.shape) > 1:
        return arr.ravel()
    return arr


def compute_r2(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Compute coefficient of determination between `y_true` and `y_pred`.

    Computed as pearson's rho ** 2
    """
    return np.corrcoef(_ensure1d(y_true), _ensure1d(y_pred))[1, 0] ** 2


def compute_pseudo_r2(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Compute pseudo-r2 between `y_true` and `y_pred`.

    Computed as SSR / SST
    """
    return r2_score(_ensure1d(y_true), _ensure1d(y_pred))


def compute_mse(
    y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Compute the mean squared error between `y_true` and `y_pred`."""

    if isinstance(y_true, torch.Tensor):
        return torch.mean((y_true - y_pred) ** 2)
    else:
        return np.mean((y_true - y_pred) ** 2)


def load_pickle(path: PosixPath) -> object:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj


def get_model_base(model: torch.nn.Module) -> torch.nn.Module:
    """Get the base for `model`. Works for RoBERTa- or BERT-based
    models

    Args:
        model (torch.nn.Module): Huggingface Model

    Returns:
        torch.nn.Module: Base roberta or bert encoder.
    """
    model_name = type(model).__name__
    if "Roberta" in model_name:
        return model.roberta
    elif "Bert" in model_name:
        return model.bert
    else:
        assert False, "Model must be based on RoBERTa or BERT."


def freeze_module(module: torch.nn.Module):
    """Freeze all parameters in pytorch module (inplace)

    Args:
        module (torch.nn.Module): module to be frozen
    """
    for p in module.parameters():
        p.requires_grad = False


def freeze_base(model: torch.nn.Module):
    """Freeze the pretrained model base.

    Args:
        model (torch.nn.Module): model whose base will be frozen.
    """
    base = get_model_base(model)
    freeze_module(base)


def get_species_type(species: str) -> str:
    """Get the type of species (coarse categories) from the ebi.ac.uk API

    Args:
        species (str): scientific genus/species name.

    Returns:
        str: one of 'non-plant', 'non-embryophyte', 'eudicot', 'monocot', 'unknown'.
    """
    try:
        genus, species = species.split("_")[:2]
    except ValueError:
        genus = ""
    name = "%20".join((genus.capitalize(), species))
    endpoint = f"https://www.ebi.ac.uk/ena/taxonomy/rest/scientific-name/{name}"
    try:
        resp = requests.get(endpoint)
        lineage = resp.json()[0]["lineage"]
        if "Viridiplantae" not in lineage:
            return "non-plant"
        if "Embryophyta" not in lineage:
            return "non-embryophyte"
        if "eudicot" in lineage:
            return "eudicot"
        return "monocot"

    except:
        return "unknown"
