"""
Fine-tuning the transformer model on the downstream gene expression prediction task
"""
import os

import torch
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from florabert import config, utils, training, dataio
from florabert import transformers as tr


DATA_DIR = config.data_final / "transformer" / "genex" / "nam"
TRAIN_DATA = "train.tsv"
EVAL_DATA = "eval.tsv"
TEST_DATA = "test.tsv"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"
PREPROCESSOR = config.models / "preprocessor" / "preprocessor.pkl"

# Starting from last checkpoint of the general purpose model
PRETRAINED_MODEL = (
    config.models / "transformer" / "language-model-finetuned" / "checkpoint-3100"
)
OUTPUT_DIR = config.models / "transformer" / "prediction-model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_search_space(trial) -> dict:
    return {
        "learning_rate": tune.loguniform(1e-5, 1e-4),
        "num_train_epochs": tune.choice(range(1, 20)),
        "seed": tune.choice(range(1, 41)),
        # "per_device_train_batch_size": 64,
        # "delay_size": tune.choice(range(0, 750)),
        # "betas": [0.99, 0.999],
        # "eps": 1e-8,
        # # "weight_decay": tune.uniform(0, 1),
        # "weight_decay": 0,
        # "warmup_steps": tune.choice(range(0, 200)),
        # "num_param_groups": 2,
    }


def load_model(args, settings):
    return tr.load_model(
        args.model_name,
        args.tokenizer_dir,
        pretrained_model=args.pretrained_model,
        log_offset=args.log_offset,
        **settings,
    )


def main():
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data=TRAIN_DATA,
        eval_data=EVAL_DATA,
        test_data=TEST_DATA,
        output_dir=OUTPUT_DIR,
        pretrained_model=PRETRAINED_MODEL,
        tokenizer_dir=TOKENIZER_DIR,
        model_name="roberta-pred-mean-pool",
        log_offset=1,
        preprocessor=PREPROCESSOR,
        transformation="log",
        hyperparam_search_metrics="mse",
        hyperparam_search_trials=10,
    )

    settings = utils.get_model_settings(config.settings, args)

    print(f"Model settings: {settings}")

    print("Making model")
    config_obj, tokenizer, model = load_model(args, settings)

    if args.freeze_base:
        print("Freezing base")
        utils.freeze_base(model)

    num_params = utils.count_model_parameters(model, trainable_only=True)
    print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")
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
        nshards=args.nshards,
    )
    dataset_train = datasets["train"]
    dataset_eval = datasets["eval"]
    dataset_test = datasets["test"]
    print(f"Loaded training data with {len(dataset_train)} examples")

    if args.nshards_eval:
        print(f"Keeping shard 1/{args.nshards_eval} of eval data")
        dataset_eval = dataset_eval.shard(num_shards=args.nshards_eval, index=1)

    data_collator = dataio.load_data_collator("pred")
    training_settings = config.settings["training"]["finetune"]
    if args.learning_rate is not None:
        training_settings["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        training_settings["num_train_epochs"] = args.num_train_epochs

    print(training_settings)

    model_init = lambda: load_model(args, settings)[2]  # For hyperparameter search

    trainer = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_eval,
        args.output_dir,
        hyperparameter_search=args.hyperparameter_search,
        model_init=model_init,
        metrics=args.metrics,
        **training_settings,
    )

    if args.hyperparameter_search:
        mode = "min" if args.direction == "minimize" else "max"
        if args.search_metric:
            print(f"Objective is {args.search_metric}")

            def compute_objective(metrics):
                return metrics[args.search_metric]

        else:
            compute_objective = None

        print(f"Searching for best hyperparameters on {torch.cuda.device_count()} GPUs")
        best_run = trainer.hyperparameter_search(
            n_trials=args.ntrials,
            direction=args.direction,
            backend="ray",
            search_alg=HyperOptSearch(mode=mode),
            scheduler=AsyncHyperBandScheduler(mode=mode),
            hp_space=make_search_space,
            n_jobs=1,
            compute_objective=compute_objective,
        )
        print(best_run)
    else:
        print(f"Starting training on {torch.cuda.device_count()} GPUs")
        training.do_training(trainer, args, args.output_dir)

    print("Final evaluation")
    metrics = trainer.evaluate(dataset_test)
    print(metrics)

    print("Saving model")
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
