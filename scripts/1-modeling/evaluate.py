"""Performance of FLORABERT overall and disaggregated by tissue.
"""
import torch
from datasets import Dataset
import pandas as pd
import numpy as np

from florabert import config, utils, metrics, dataio
from florabert import transformers as tr
from florabert import visualization as vis


TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
PRETRAINED_MODEL = config.models / "transformer" / "full-dataset"
DATA_DIR = config.data_final / "transformer" / "genex"

DATA_DIR = config.data_final / "transformer" / "genex" / "nam"
TEST_DATA = "test.tsv"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
PREPROCESSOR = config.models / "preprocessor" / "preprocessor.pkl"
OUTPUT_DIR = config.output / "transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(args, settings):
    return tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        log_offset=args.log_offset,
        **settings,
    )


def load_data(tokenizer, test_data) -> Dataset:
    datasets = dataio.load_datasets(
        tokenizer, test_data, seq_key="sequence", file_type="csv", delimiter="\t"
    )
    return datasets["train"]


def get_metrics(num_tissues) -> tuple:
    metric_types = ["mse", "mae", "r2", "pseudo-r2"]
    metric_names = [
        *[
            f"{metric}_{config.tissues[i]}"
            for metric in metric_types
            for i in range(num_tissues)
        ],
        *metric_types,
    ]
    metric_fns = [
        *[
            metrics.make_tissue_loss(i, metric=metric)
            for metric in metric_types
            for i in range(num_tissues)
        ],
        torch.nn.MSELoss(),
        metrics.make_mae_loss(),
        utils.compute_r2,
        utils.compute_pseudo_r2,
    ]
    return metric_names, metric_fns


def package_metrics(results, metric_names) -> pd.DataFrame:
    data = {}
    for metric, value in zip(metric_names, results):
        name_split = metric.split("_")
        if len(name_split) == 1:
            metric_name = metric
            tissue = "all"
        else:
            metric_name, tissue = name_split
        tissue_names = data.get("tissue") or []
        if tissue not in tissue_names:
            tissue_names.append(tissue)
            data["tissue"] = tissue_names
        metric_values = data.get(metric_name) or []
        metric_values.append(value)
        data[metric_name] = metric_values
    return pd.DataFrame(data)


def main():
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data=TEST_DATA,
        test_data=TEST_DATA,
        output_dir=OUTPUT_DIR,
        pretrained_model=PRETRAINED_MODEL,
        tokenizer_dir=TOKENIZER_DIR,
        model_name="roberta-pred-mean-pool",
        log_offset=1,
        preprocessor=PREPROCESSOR,
        transformation="log",
    )

    settings = utils.get_model_settings(config.settings, args)

    if args.output_mode:
        settings["output_mode"] = args.output_mode
    if args.tissue_subset is not None:
        settings["num_labels"] = len(args.tissue_subset)

    print(f"Model settings: {settings}")

    print("Making model")
    config_obj, tokenizer, model = load_model(args, settings)

    print("Loading data")
    preprocessor = utils.load_pickle(args.preprocessor) if args.preprocessor else None

    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        eval_data=args.eval_data,
        test_data=args.test_data,
        seq_key="sequence",
        file_type="csv",
        delimiter="\t",
        log_offset=args.log_offset,
        preprocessor=preprocessor,
        filter_empty=args.filter_empty,
        tissue_subset=args.tissue_subset,
        threshold=args.threshold,
        transformation=args.transformation,
        discretize=(args.output_mode == "classification"),
        shuffle=False,
    )
    dataset_test = datasets["train"]

    print("Getting predictions")
    trg_true, trg_pred = metrics.get_predictions(model, dataset_test)

    metric_names, metric_fns = get_metrics(settings["num_labels"])

    print("Evaluating")
    results = metrics.evaluate_model(trg_true, trg_pred, metric_fns)
    df = package_metrics(results, metric_names)
    utils.save_model_performance(df, "florabert")

    print("Creating scatterplot")
    vis.scatter_genex_predictions(
        trg_true, trg_pred, args.output_dir, args.tissue_subset
    )

    print("Loading maize lines")
    maize_lines = pd.read_csv(
        config.data_final / "nam_data" / "merged_seq_genex.csv"
    ).loc[lambda df: df["fold"] == "test", ["gene_id"]]

    nam_metadata = pd.read_csv(config.data_final / "nam_data" / "nam_metadata.csv")

    maize_lines["gene_prefix"] = maize_lines["gene_id"].str.slice(0, 9)
    maize_lines_joined = maize_lines.merge(nam_metadata, on="gene_prefix", how="left")

    assert (
        maize_lines["gene_id"].values == maize_lines_joined["gene_id"].values
    ).all(), "After joining the indices are not aligned."

    print("Evaluating for each maize line")

    cultivar_results = []
    for cultivar in nam_metadata["cultivar"]:
        print(cultivar)
        flag = (maize_lines_joined["cultivar"] == cultivar).values  # type: np.ndarray
        if flag.sum() == 0:
            print(f"No test genes for {cultivar}. Skipping.")
            continue
        cultivar_true, cultivar_pred = trg_true[flag], trg_pred[flag]
        results = metrics.evaluate_model(cultivar_true, cultivar_pred, metric_fns)
        df = package_metrics(results, metric_names)
        df = df.assign(cultivar=cultivar)
        cultivar_results.append(df)

    cultivar_results = pd.concat(cultivar_results, axis=0)
    utils.save_model_performance(cultivar_results, "nam_lines")


if __name__ == "__main__":
    main()
