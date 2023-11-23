"""
File: financial_features_longformer.py
Author: Dimitrios Mamakas (Athens University of Economics and Business)
Date: November 23, 2023
Description: Implementation of the following Longformer variants
                • ipo-longformer-txff-cls-8192
                • ipo-longformer-txff-cls-20480
                • ipo-longformer-extra-global-txff-cls-8192
                • ipo-longformer-extra-global-txff-cls-20480


License:
This code is provided under the MIT License.
You are free to copy, modify, and distribute the code.
If you use this code in your research, please include a reference to the original study (please visit the home page).
"""

import torch
import logging
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import LongformerForSequenceClassification, LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerClassificationHead, \
    LongformerSequenceClassifierOutput, LongformerEncoder, LongformerEmbeddings, LongformerPooler, \
    LongformerBaseModelOutputWithPooling

""""
Logger instance.
"""
logger = logging.getLogger(__name__)


class CustomFinancialLongformerClassificationHead(LongformerClassificationHead):
    def __init__(self, config):
        super().__init__(config)
        # Pre-defined dense layer & dropout
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Custom two dense layers
        self.first_dense_layer = nn.Linear(config.hidden_size, config.reduction_features)
        self.second_dense_layer = nn.Linear(config.reduction_features + 8, (config.reduction_features + 8) // 2)

        # Output layer
        self.out_proj = nn.Linear((config.reduction_features + 8) // 2, config.num_labels)
        self.out_proj_ = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, financial_features, **kwargs):
        # Take <s> token (equiv. to [CLS])
        hidden_states = hidden_states[:, 0, :]

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if financial_features is not None:
            # Reduce the hidden states
            reduced_hidden_states = self.first_dense_layer(hidden_states)

            # Perform the concatenation with the equivalent financial embedding
            concatenated_embedding = torch.cat((reduced_hidden_states, financial_features), dim=1)

            # Reduce the concatenated embedding
            reduced_concatenated_embedding = self.second_dense_layer(concatenated_embedding)

            output = self.out_proj(reduced_concatenated_embedding)
        else:
            output = self.out_proj_(hidden_states)

        return output


class FinancialLongformerModel(LongformerModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            financial_features=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[
                                                :, 0, 0, :
                                                ]

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LongformerBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            global_attentions=encoder_outputs.global_attentions,
        )


class LongformerFinancialModelForSequenceClassification(LongformerForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.longformer = FinancialLongformerModel(config, add_pooling_layer=False)

        # Custom two dense Layers
        self.first_dense_layer = nn.Linear(config.hidden_size, config.reduction_features)
        self.second_dense_layer = nn.Linear(config.reduction_features + 8, (config.reduction_features + 8) // 2)

        # Classifier
        self.classifier = CustomFinancialLongformerClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        financial_features=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            financial_features=financial_features,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # Output
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, financial_features)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

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

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )