"""
Modified HuggingFace transformer model classes
"""
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss, PoissonNLLLoss, KLDivLoss

from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import RobertaPreTrainedModel


class RobertaMeanPoolConfig(RobertaConfig):
    model_type = "roberta"

    def __init__(
        self,
        output_mode="regression",
        freeze_base=True,
        start_token_idx=0,
        end_token_idx=1,
        threshold=1,
        alpha=0.5,
        log_offset=1,
        batch_norm=False,
        **kwargs,
    ):
        """Constructs RobertaConfig."""
        super().__init__(**kwargs)
        self.output_mode = output_mode
        self.freeze_base = freeze_base
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.threshold = threshold
        self.alpha = alpha
        self.log_offset = log_offset
        self.batch_norm = batch_norm


class ClassificationHeadMeanPool(nn.Module):
    """Head for sentence-level classification tasks.

    Modifications:
        1. Using mean-pooling over tokens instead of CLS token
        2. Multi-output regression
    """

    def __init__(self, config: RobertaMeanPoolConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.start_token_idx = config.start_token_idx
        self.end_token_idx = config.end_token_idx
        self.batch_norm = (
            nn.BatchNorm1d(config.hidden_size) if config.batch_norm else None
        )
        if self.batch_norm is not None:
            print("Using batch_norm")

    def forward(self, features, attention_mask=None, input_ids=None, **kwargs):
        x = self.embed(features, attention_mask, input_ids, **kwargs)
        x = self.out_proj(x)
        return x

    def embed(self, features, attention_mask=None, input_ids=None, **kwargs):
        attention_mask[input_ids == self.start_token_idx] = 0
        attention_mask[input_ids == self.end_token_idx] = 0
        x = torch.sum(features * attention_mask.unsqueeze(2), dim=1) / torch.sum(
            attention_mask, dim=1, keepdim=True
        )  # Mean pooling over non-padding tokens

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        # Batchnorm
        x = self.normalize(x)

        # Second linear layer
        x = self.dense2(x)
        x = torch.tanh(x)
        return x

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_norm is not None:
            return self.batch_norm(x)
        return x


class ClassificationHeadMeanPoolSparse(nn.Module):
    """Classification head that predicts binary outcome (expressed/not)
    and real-valued gene expression values.
    """

    def __init__(self, config):
        super().__init__()
        self.classification_head = ClassificationHeadMeanPool(config)
        self.regression_head = ClassificationHeadMeanPool(config)

    def forward(
        self, features, attention_mask=None, input_ids=None, **kwargs
    ) -> Tuple[torch.Tensor]:
        """Compute binarized logits and real-valued gene expressions for each tissue.

        Args:
            features (torch.Tensor): outputs of RoBERTa
            attention_mask (Optional[torch.Tensor]): attention mask for sentence
            input_ids (Optional[torch.Tensor]): original sequence inputs

        Returns:
            (torch.Tensor): classification logits (whether gene is expressed/not for tissue)
            (torch.Tensor): gene expression value predictions (real-valued)
        """
        # Consider using .clone().detach()
        attention_mask_copy = attention_mask.clone()
        return (
            self.classification_head(
                features, attention_mask=attention_mask, input_ids=input_ids, **kwargs
            ),
            self.regression_head(
                features,
                attention_mask=attention_mask_copy,
                input_ids=input_ids,
                **kwargs,
            ),
        )


class SparseMSELoss(nn.Module):
    """Custom loss function that takes in two inputs:
    1. Predicted logits for whether gene is expressed (1) or not (0)
    2. Real-valued log-TPM values for gene expression predictions.
    """

    def __init__(self, threshold: float = 1, alpha: float = 0.5):
        """
        Args:
            threshold (float): any value below this threshold (in natural
                scale, NOT log-scale) is considered "not expressed"
            alpha (float): parameter controlling importance of classification
                in overall accuracy. alpha == 1 means this is identical to
                classification. alpha == 0 means this is identical to regression.
        """
        super().__init__()
        self.threshold = np.log(threshold)
        self.alpha = alpha
        self.mse = MSELoss()
        self.bce = BCEWithLogitsLoss()

    def forward(self, logits: Tuple[torch.Tensor], labels: torch.Tensor):
        classification_outputs, regression_outputs = logits
        binarized_labels = (labels >= self.threshold).float()

        mse_loss = self.mse(regression_outputs, labels)
        bce_loss = self.bce(classification_outputs, binarized_labels)

        # Weight the losses by the logits
        # the mse loss should be weighted by the probability of being expressed
        # the bce loss should be weighted by the probability of not being expressed

        loss = self.alpha * bce_loss + (1 - self.alpha) * mse_loss
        return loss


class ZeroInflatedNegativeBinomialNLL(nn.Module):
    """Custom loss function that calculates the negative log-likelihood
    according to a zero-inflated negative binomial model.
    """

    pass


# -------------------------------------- #
#                                        #
# ---------- Modified RoBERTa ---------- #
#                                        #
# -------------------------------------- #


class RobertaForSequenceClassificationMeanPool(RobertaPreTrainedModel):
    """RobertaForSequenceClassification using modified classification head

    Args:
        RobertaPreTrainedModel ([type]): [description]

    Returns:
        [type]: [description]
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaMeanPoolConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.output_mode = config.output_mode or "regression"
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.threshold = config.threshold
        self.alpha = config.alpha
        self.log_offset = config.log_offset

        if self.output_mode == "sparse":
            self.classifier = ClassificationHeadMeanPoolSparse(config)
        else:
            self.classifier = ClassificationHeadMeanPool(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(
            sequence_output, attention_mask=attention_mask, input_ids=input_ids
        )

        loss = None
        if labels is not None:
            if self.output_mode == "regression":
                loss_fct = MSELoss()
            elif self.output_mode == "sparse":
                loss_fct = SparseMSELoss(threshold=self.threshold, alpha=self.alpha)
            elif self.output_mode == "classification":
                loss_fct = BCEWithLogitsLoss()
            elif self.output_mode == "poisson":
                loss_fct = PoissonNLLLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Embed sequences by running the `forward` method up to the dense layer of the classifier"""
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        embeddings = self.classifier.embed(
            sequence_output, attention_mask=attention_mask, input_ids=input_ids
        )
        return embeddings

    def get_tissue_embeddings(self):
        return self.classifier.out_proj.weight.detach()

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        if self.output_mode == "sparse":
            binary_logits, pred_values = logits
            # Convert logits to binary predictions
            binary_preds = binary_logits < 0
            # return binary_preds * pred_values
            pred_values[binary_preds] = np.log(self.log_offset)
            return pred_values
        return logits


# -------------------------------------- #
#                                        #
# ----------  Modified BERT  ----------- #
#                                        #
# -------------------------------------- #


class BertMeanPoolConfig(BertConfig):
    model_type = "bert"

    def __init__(
        self, output_mode="regression", start_token_idx=2, end_token_idx=3, **kwargs
    ):
        """Constructs BertConfig."""
        super().__init__(**kwargs)
        self.output_mode = output_mode
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx


class BertForSequenceClassificationMeanPool(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.output_mode = config.output_mode or "regression"
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = ClassificationHeadMeanPool(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(
            pooled_output, attention_mask=attention_mask, input_ids=input_ids
        )

        loss = None
        if labels is not None:
            if self.output_mode == "regression":
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
