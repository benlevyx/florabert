"""
training.py

Functions and classes for training pytorch models.
"""
import os
from pathlib import PosixPath
from typing import Callable, Union
import multiprocessing as mp
import inspect

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam, AdamW
from torch.optim.optimizer import Optimizer
from transformers import (
    TrainingArguments,
    Trainer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    PreTrainedTokenizerFast,
)
from transformers.integrations import WandbCallback
from datasets import Dataset
from dataclasses import dataclass, field

from torch_optimizer import Lamb
from transformers.utils.dummy_pt_objects import (
    PreTrainedModel,
)

from .utils import get_latest_checkpoint, compute_r2, compute_mse


def make_constant_schedule(
    optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1
) -> Callable:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def make_lr_lambda_with_delay(
    num_warmup_steps: int,
    num_training_steps: int,
    num_delay_steps: int = 0,
    max_level: float = 1.0,
) -> Callable:
    """Make a learning rate lambda function with a delay (for model fine-tuning)"""
    actual_training_steps = num_training_steps - num_delay_steps

    def lr_lambda(current_step: int):
        step = current_step - num_delay_steps
        if step < 0:
            lr = 0.0
        elif step < num_warmup_steps:
            lr = float(step) / float(max(1, num_warmup_steps))
        else:
            # lr = max(
            #     0.0,
            #     (
            #         float(actual_training_steps - step)
            #         / float(max(1, actual_training_steps - num_warmup_steps))
            #     ),
            # )
            # Warmup + constant
            lr = 1.0
        return lr * max_level

    return lr_lambda


def get_plateau_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_cooldown_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> Callable:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps - num_cooldown_steps:
            return (1 / num_cooldown_steps) * (num_training_steps - current_step)
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_optimizer(optimizer, model, num_param_groups, param_group_size, **kwargs):
    if num_param_groups or param_group_size:
        param_groups = make_param_groups(model, num_param_groups, param_group_size)
    else:
        param_groups = model.parameters()
    if "learning_rate" in kwargs:
        kwargs["lr"] = kwargs.pop("learning_rate")
    if optimizer == "lamb":
        return Lamb(param_groups, **kwargs)
    elif optimizer == "adam":
        return Adam(param_groups, **kwargs)
    elif optimizer == "adamw":
        return AdamW(param_groups, **kwargs)
    elif callable(optimizer):
        return optimizer(param_groups, **kwargs)
    else:
        raise ValueError(f"Unrecognized optimizer type: {optimizer}")


def _get_scheduler(
    scheduler,
    optimizer,
    num_training_steps,
    num_warmup_steps: int = None,
    num_param_groups: int = None,
    num_cooldown_steps: int = None,
    delay_size: int = None,
):
    if scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
        # return make_constant_schedule(optimizer, num_warmup_steps)
    elif scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler == "plateau":
        num_cooldown_steps = num_cooldown_steps or num_warmup_steps
        return get_plateau_schedule_with_warmup(
            optimizer, num_warmup_steps, num_cooldown_steps, num_training_steps
        )
    elif scheduler == "cosine_with_hard_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles=4
        )
    elif scheduler == "delay":
        lr_lambdas = [
            make_lr_lambda_with_delay(
                num_warmup_steps, num_training_steps, i * delay_size, 0.5 ** i
            )  # 0.5 ** i denotes a max lr of 1.0, 0.5, 0.25, 0.125 x the overall lr
            for i in range(num_param_groups)
        ]
        # We reverse `lr_lambdas` because the largest delays should go to the lower layers
        # while the highest layers should have little or no delay
        return LambdaLR(optimizer, lr_lambda=lr_lambdas[::-1])


def make_param_groups(
    model: PreTrainedModel, num_param_groups: int = None, param_group_size: int = None
):
    """
    Separating out the parameter groups for delayed learning
    This creates 4 parameter groups
    1. RoBERTa layers 1 + 2
    2. RoBERTa layers 3 + 4
    3. RoBERTa layers 5 + 6
    4. RoBERTa classification head

    These will be trained with delayed linear warmup, with
    (4) training with no delay, (3) training with delay 500,
    (2) training with delay 1000, and (1) training with delay 1500
    """
    # This is still kinda hacky
    model_name = type(model).__name__
    if "Roberta" in model_name:
        layers = model.roberta.encoder.layer
    else:
        layers = model.bert.encoder.layer

    num_encoder_layers = len(layers)

    if num_param_groups:
        if not param_group_size:
            param_group_size = int(num_encoder_layers // num_param_groups)
    if not num_param_groups:
        if not param_group_size:
            # The default
            param_group_size = 2
        num_param_groups = int(num_encoder_layers // param_group_size)

    print(f"Number of encoder layers: {num_encoder_layers}")
    print(f"Number of parameter groups: {num_param_groups}")
    print(f"Parameter group size: {param_group_size}")

    param_groups = []
    for i in range(num_param_groups):
        params = []
        for j in range(param_group_size):
            params.extend(layers[i * param_group_size + j].parameters())
        param_groups.append(dict(params=params))
    param_groups.append(dict(params=list(model.classifier.parameters())[0]))

    print(len(param_groups))

    return param_groups


def make_trainer(
    model: torch.nn.Module,
    data_collator: callable,
    train_dataset: Dataset,
    test_dataset: Dataset,
    output_dir: Union[str, PosixPath],
    overwrite_output_dir: bool = True,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None,
    delay_size: int = None,
    hyperparameter_search: bool = False,
    model_init: callable = None,
    metrics: Union[list, str] = None,
    **training_kwargs,
) -> Trainer:
    """Make a Huggingface transformers Trainer.

    Args:
        model (torch.nn.Module): model to be trained
        data_collator (callable): data collation function
        train_dataset (Dataset): training dataset
        test_dataset (Dataset): test/evaluation dataset
        output_dir (Union[str, PosixPath]): directory where model and
            checkpoints will be saved
        overwrite_output_dir (bool, optional): whether to overwrite files
            saved in `output_dir`. Defaults to True.
        optimizer (torch.optim.Optimizer, optional): optimizer to use. If none,
            then defaults to AdamW. Defaults to None.
        scheduler (torch.optim.lr_scheduler.LambdaLR, optional): utility to
            adjust the learning rate. Defaults to None.
        param_groups (list, optional): groups of parameters to optimize
            differently. Defaults to None.
        delay_size (int, optional): if using delayed scheduling, how long to
            delay learning between parameter groups. Defaults to None.
        hyperparameter_search (bool, optional): whether to do hyperparameter
            search. Defaults to False.
        model_init (callable, optional): model initialization function if doing
            hyperparameter search. Defaults to None.
        metrics (Union[list, str], optional): metrics to evaluate hyperparameter
            search trials on. Defaults to 'r2'.
    Returns:
        Trainer: [description]
    """
    training_args = MyTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=overwrite_output_dir,
        evaluation_strategy="steps",
        # TODO: Figure out which setting for logging R2
        prediction_loss_only=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        evaluate_during_training=True,
        do_eval=True,
        **training_kwargs,
    )

    optimizers = None
    if optimizer is not None:
        assert (
            scheduler is not None
        ), "If optimizer is not None, a scheduler must be supplied"
        num_devices = 1 if not torch.cuda.is_available() else torch.cuda.device_count()
        num_training_steps = np.floor(
            len(train_dataset)
            / training_kwargs["per_device_train_batch_size"]
            * training_kwargs["num_train_epochs"]
            / training_kwargs["gradient_accumulation_steps"]
            / num_devices
        )

        def create_optimizer_and_scheduler(
            num_training_steps,
            num_param_groups=None,
            param_group_size=None,
            opt_kwargs=None,
            sched_kwargs=None,
            **kwargs,
        ):
            opt_kwargs = opt_kwargs if opt_kwargs is not None else {}
            sched_kwargs = sched_kwargs if sched_kwargs is not None else {}

            opt = _get_optimizer(
                optimizer,
                model,
                num_param_groups=num_param_groups,
                param_group_size=param_group_size,
                **opt_kwargs,
            )
            if num_param_groups is None:
                num_param_groups = 0
            sched = _get_scheduler(
                scheduler,
                opt,
                num_training_steps,
                num_param_groups=num_param_groups + 1,
                **sched_kwargs,
            )
            return (opt, sched)

    else:
        create_optimizer_and_scheduler = None

    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is None:
        metrics = []

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            # logits, hidden_states = predictions
            predictions, _ = predictions
        res = {}
        if "r2" in metrics:
            res["r2"] = compute_r2(labels, predictions)
        if "mse" in metrics:
            res["mse"] = compute_mse(labels, predictions)
        return res

    if hyperparameter_search:
        return make_searcher(
            training_args,
            model_init,
            train_dataset,
            test_dataset,
            create_optimizer_and_scheduler=create_optimizer_and_scheduler,
            delay_size=delay_size,
            compute_metrics=compute_metrics,
        )

    if create_optimizer_and_scheduler:
        opt_kwargs = {
            k: v
            for k, v in training_kwargs.items()
            if k in ["betas", "eps", "weight_decay", "learning_rate"]
        }
        sched_kwargs = {
            k: v
            for k, v in training_kwargs.items()
            if k in ["delay_size", "num_warmup_steps", "num_cooldown_steps"]
        }
        if "warmup_steps" in training_kwargs:
            sched_kwargs["num_warmup_steps"] = training_kwargs["warmup_steps"]
        if delay_size is not None:
            sched_kwargs["delay_size"] = delay_size

        print(training_kwargs)

        optimizers = create_optimizer_and_scheduler(
            num_training_steps,
            num_param_groups=training_kwargs.get("num_param_groups", None),
            opt_kwargs=opt_kwargs,
            sched_kwargs=sched_kwargs,
        )
    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
    )


def do_training(trainer, args, output_dir):
    """
    Run HuggingFace trainer, loading latest checkpoint if `args.warmstart`
    is True.
    """
    if args.warmstart:
        ckpt = get_latest_checkpoint(output_dir)
        print(f"Resuming training from {ckpt}")
        trainer.train(str(ckpt))
    else:
        trainer.train()

    return trainer


@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    delay_size (int, optional): Size of delay between subsequent blocks
    betas (tuple, optional): Adam betas
    eps (float, optional)
    weight_decay (float, optional)
    """

    delay_size: int = field(default=500)
    betas: tuple = field(default=(0.9, 0.999))
    eps: float = field(default=1e-8)
    weight_decay: float = field(default=0)
    warmup_steps: int = field(default=500)
    num_cooldown_steps: int = field(default=500)
    num_param_groups: int = field(default=2)
    param_group_size: int = field(default=None)


class MyTrainer(Trainer):
    """Subclass of huggingface transformers.Trainer to enable custom scheduler
    and optimizer for hyperparameter search.
    """

    def __init__(
        self,
        model: PreTrainedModel = None,
        args: MyTrainingArguments = None,
        create_optimizer_and_scheduler=None,
        **kwargs,
    ):
        assert (
            create_optimizer_and_scheduler
        ), "`create_optimizer_and_scheduler` must be supplied. Otherwise, use `Trainer`"
        super().__init__(model=model, args=args, **kwargs)
        self._create_optimizer_and_scheduler = create_optimizer_and_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        opt, sched = self._create_optimizer_and_scheduler(
            num_training_steps,
            num_param_groups=self.args.num_param_groups,
            param_group_size=self.args.param_group_size,
            opt_kwargs=dict(
                betas=self.args.betas,
                eps=self.args.eps,
                weight_decay=self.args.weight_decay,
            ),
            sched_kwargs=dict(
                delay_size=self.args.delay_size,
                num_warmup_steps=self.args.warmup_steps,
                num_cooldown_steps=self.args.num_cooldown_steps,
            ),
        )

        self.optimizer = opt
        self.lr_scheduler = sched


def make_searcher(
    training_args: TrainingArguments,
    model_init: Callable,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    create_optimizer_and_scheduler: callable = None,
    delay_size: int = 0,
    compute_metrics: Callable = None,
) -> Trainer:
    """Create Trainer for hyperparameter search

    Args:
        training_args (TrainingArguments): training arguments
        model_init (callable): model initialization function
        metrics (Union[str, list]): metrics to evaluate each trial
        create_optimizer_and_scheduler (callable, optional): function to create
            the optimizer and scheduler if not using default. Defaults to None.

    Returns:
        Trainer
    """
    kwargs = dict(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # if create_optimizer_and_scheduler:
    trainer = MyTrainer(
        **kwargs, create_optimizer_and_scheduler=create_optimizer_and_scheduler
    )
    # trainer = Trainer(**kwargs)
    trainer.remove_callback(WandbCallback)

    return trainer
