import torch
from torch import nn
from transformers import BertPreTrainedModel
from models.financial_features_bert import BertFinancial
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput


class FinancialBertMaxPooledForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertFinancial(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        if config.multiple_dense_layers:
            # Use two dense layers on top of each other in the architecture
            self.first_dense_layer = nn.Linear(config.hidden_size, config.reduction_features)
            self.second_dense_layer = nn.Linear(config.reduction_features + 8, (config.reduction_features + 8) // 2)
            self.classifier = nn.Linear((config.reduction_features + 8) // 2, config.num_labels)
        else:
            # Use only one dense layer in the architecture
            self.dense_layer = nn.Linear(config.hidden_size, config.reduction_features)
            self.classifier = nn.Linear(config.reduction_features + 8, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            financial_features=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            financial_features=None,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Max pooled outputs
        pooled_output = torch.max(outputs.last_hidden_state, dim=1)[0]
        pooled_output = self.dropout(pooled_output)

        if hasattr(self, 'second_dense_layer'):
            # Use the dense layer in order to reduce the BERT's output dimension from 768 to 8.
            reduced_embedding = self.first_dense_layer(pooled_output)

            # Perform the concatenation with the equivalent financial embedding
            concatenated_embedding = torch.cat((reduced_embedding, financial_features), dim=1)

            reduced_concatenated_embedding = self.second_dense_layer(concatenated_embedding)

            # Perform the classification.
            logits = self.classifier(reduced_concatenated_embedding)
        else:
            # Use the dense layer in order to reduce the BERT's output dimension from 768 to 8.
            reduced_embedding = self.dense_layer(pooled_output)

            # Perform the concatenation with the equivalent financial embedding
            concatenated_embedding = torch.cat((reduced_embedding, financial_features), dim=1)

            # Perform the classification.
            logits = self.classifier(concatenated_embedding)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
