from pathlib import PosixPath
from typing import Union, Optional

from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification
)

from .models import (
    RobertaMeanPoolConfig,
    RobertaForSequenceClassificationMeanPool,
    BertMeanPoolConfig,
    BertForSequenceClassificationMeanPool
)
from .nlp import DNABERTTokenizer


RobertaSettings = dict(
    padding_side='left'
)
DnabertSettings = dict(
    k=6,
    do_lower_case=False,
    padding_side='right'
)


MODELS = {
    "roberta-lm": (RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, RobertaSettings),
    "roberta-pred": (RobertaConfig, RobertaTokenizerFast, RobertaForSequenceClassification, RobertaSettings),
    "roberta-pred-mean-pool": (RobertaMeanPoolConfig, RobertaTokenizerFast, RobertaForSequenceClassificationMeanPool, RobertaSettings),
    "dnabert-lm": (BertConfig, DNABERTTokenizer, BertForMaskedLM, DnabertSettings),
    "dnabert-pred": (BertConfig, DNABERTTokenizer, BertForSequenceClassification, DnabertSettings),
    "dnabert-pred-mean-pool": (BertMeanPoolConfig, DNABERTTokenizer, BertForSequenceClassificationMeanPool, DnabertSettings)
}


def load_model(model_name: str,
               tokenizer_dir: Union[str, PosixPath],
               max_tokenized_len: int = 254,
               pretrained_model: Union[str, PosixPath] = None,
               k: Optional[int] = None,
               do_lower_case: Optional[bool] = None,
               padding_side: Optional[str] = 'left',
               **config_settings) -> tuple:
    """Load specified model, config, and tokenizer.

    Args:
        model_name (str): Name of model. Acceptable options are
            - 'roberta-lm',
            - 'roberta-pred',
            - 'roberta-pred-mean-pool'
            - 'dnabert'
            - 'dnabert-pred'
            - 'dnabert-pred-mean-pool'
        tokenizer_dir (Union[str, PosixPath]): Directory containing tokenizer
            files: merges.txt and vocab.txt
        max_len (int, optional): Maximum tokenized length,
            not including SOS and EOS. Defaults to 254.
        pretrained_model (Union[str, PosixPath], optional): path to saved
            pretrained RoBERTa transformer model. Defaults to None.
        k (Optional[int], optional): Size of kmers (for DNABERT model). Defaults to 6.
        do_lower_case (bool, optional): Whether to convert all inputs to lower case. Defaults to None.
        padding_side (str, optional): Which side to pad on. Defaults to 'left'.

    Returns:
        tuple: config_obj, tokenizer, model
    """
    config_settings = config_settings or {}
    max_position_embeddings = max_tokenized_len + 2  # To include SOS and EOS
    config_class, tokenizer_class, model_class, tokenizer_settings = MODELS[model_name]
    
    kwargs = dict(
        max_len=max_tokenized_len,
        truncate=True,
        padding="max_length",
        **tokenizer_settings
    )
    if k is not None:
        kwargs.update(dict(k=k))
    if do_lower_case is not None:
        kwargs.update(dict(do_lower_case=do_lower_case))
    if padding_side is not None:
        kwargs.update(dict(padding_side=padding_side))

    tokenizer = tokenizer_class.from_pretrained(str(tokenizer_dir), **kwargs)
    name_or_path = str(pretrained_model) or ''
    config_obj = config_class(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_position_embeddings,
        name_or_path=name_or_path,
        output_hidden_states=True,
        **config_settings
    )
    if pretrained_model:
        print(f"Loading from pretrained model {pretrained_model}")
        model = model_class.from_pretrained(
            str(pretrained_model), config=config_obj)
    else:
        print("Loading untrained model")
        model = model_class(config=config_obj)
    model.resize_token_embeddings(len(tokenizer))
    return config_obj, tokenizer, model
