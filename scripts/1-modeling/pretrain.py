"""
Pretraining on masked language model task.
"""
import torch

from florabert import config, utils, training, dataio
from florabert import transformers as tr

DATA_DIR = config.data_final / "transformer" / "seq"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"

OUTPUT_DIR = config.models / "transformer" / "language-model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data="all_seqs_train.txt",
        test_data="all_seqs_test.txt",
        output_dir=OUTPUT_DIR,
        model_name="roberta-lm",
    )

    settings = utils.get_model_settings(config.settings, args.model_name)

    print(args)

    config_obj, tokenizer, model = tr.load_model(
        args.model_name,
        TOKENIZER_DIR,
        pretrained_model=args.pretrained_model,
        **settings,
    )

    num_params = utils.count_model_parameters(model, trainable_only=True)
    print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")

    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        test_data=args.test_data,
        file_type="text",
        seq_key="text",
    )
    dataset_train = datasets["train"]
    dataset_test = datasets["test"]
    print(f"Loaded training data with {len(dataset_train):,} examples")
    data_collator = dataio.load_data_collator(
        "language-model",
        tokenizer=tokenizer,
        mlm_prob=config.settings["training"]["pretrain"].pop("mlm_prob"),
    )

    training_settings = config.settings["training"]["pretrain"]
    trainer = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_test,
        args.output_dir,
        **training_settings,
    )

    print(f"Starting training on {torch.cuda.device_count()} GPUs")
    training.do_training(trainer, args, args.output_dir)

    print("Saving model")

    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
