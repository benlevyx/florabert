import pytest

from torch.optim import Adam

from florabert import config, utils
from florabert import transformers as tr


# Helper functions
def load_roberta_model(**kwargs):
    return tr.load_model(
        "roberta-pred-mean-pool", config.models / "byte-level-bpe-tokenizer", **kwargs
    )


# Tests
def test_roberta_mean_pool_load_new():
    try:
        load_roberta_model()
    except:
        raise Exception(
            "Failed to load new RobertaForSequenceClassificationMeanPool model"
        )


def test_roberta_mean_pool_load_pretrained():
    try:
        load_roberta_model(
            pretrained_model=config.models / "transformer" / "language-model"
        )
    except:
        raise Exception(
            "Failed to load pretrained RobertaForSequenceClassificationMeanPool model"
        )


def test_get_lamb_optimizer():
    _, _, model = load_roberta_model()
    optimizer = tr._get_optimizer("lamb", model)
    assert optimizer is not None, "Failed to load optimizer"


def test_linear_scheduler():
    tr._get_scheduler("linear", Adam(), 10000, num_warmup_steps=500)


def test_delay_scheduler():
    tr._get_scheduler(
        "delay", Adam(), 10000, num_warmup_steps=500, num_param_groups=4, delay_size=400
    )


def test_make_trainer_simple():
    pass


def test_make_trainer_delay():
    pass


def test_load_datasets():
    pass


def test_convert_str_to_list():
    data = ["[32.0, 430.5]", "[20.0, 0.01]", "[-1419, 4]"]
    lists = tr.convert_str_to_list(data)

    assert type(lists) == list, "Result is not a list"
    assert all([type(li) == list] for li in lists), "Inner lists are not lists"
    assert all(
        [all([type(d) == float for d in li]) for li in lists]
    ), "Elements are not float"
