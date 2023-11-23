"""
File: financial_features_hierarchical_bert.py
Author: Dimitrios Mamakas (Athens University of Economics and Business)
Date: November 23, 2023
Description: Implementation of the following Hierarchical-BERT variants
                • hierbert-txff-cls-8192
                • hierbert-txff-cls-20480


License:
This code is provided under the MIT License.
You are free to copy, modify, and distribute the code.
If you use this code in your research, please include a reference to the original study (please visit the home page).
"""

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from models.financial_features_bert import BertFinancial
from models.hierarchical_bert import sinusoidal_init, SimpleOutput


class HierarchicalBertFinancialModelForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertFinancial(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear((config.reduction_features + 8) // 2, config.num_labels)

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            financial_features=financial_features,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Perform the classification.
        logits = self.classifier(outputs.last_hidden_state)

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


class FinancialHierarchicalBert(nn.Module):

    def __init__(self, config, encoder, max_segments=64, max_segment_length=128, max_pooled=False):
        super(FinancialHierarchicalBert, self).__init__()
        supported_models = ['bert', 'roberta', 'deberta']
        assert encoder.config.model_type in supported_models  # other model types are not supported so far

        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder

        # Max-Pooling variant
        self.max_pooled = max_pooled

        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length

        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments + 1, encoder.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments + 1, encoder.config.hidden_size))

        self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size,
                                          nhead=encoder.config.num_attention_heads,
                                          batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                          activation=encoder.config.hidden_act,
                                          dropout=encoder.config.hidden_dropout_prob,
                                          layer_norm_eps=encoder.config.layer_norm_eps,
                                          num_encoder_layers=2, num_decoder_layers=0).encoder

        # Add the two dense layers on top of the architecture
        self.first_dense_layer = nn.Linear(config.hidden_size, config.reduction_features)
        self.second_dense_layer = nn.Linear(config.reduction_features + 8, (config.reduction_features + 8) // 2)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                financial_features=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None

        # Encode segments with BERT --> (256, 128, 768)
        encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                       attention_mask=attention_mask_reshape,
                                       token_type_ids=token_type_ids_reshape)[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        if self.max_pooled:
            # Gather the maximum element from each vector of each segment
            encoder_outputs, _ = torch.max(encoder_outputs, dim=3)  # Size -> (4, 64, 128)

            batch_size = encoder_outputs.size()[0]

            # Reshape tensor to (n * 64, 128)
            encoder_outputs = encoder_outputs.view(-1, self.max_segment_length)

            # Linear transformation to (n * 64, 768)
            linear = nn.Linear(self.max_segment_length, self.encoder.config.hidden_size).to('cuda')
            encoder_outputs = linear(encoder_outputs)

            # Reshape transformed tensor back to (n, 64, 768)
            encoder_outputs = encoder_outputs.view(batch_size, self.max_segments, self.encoder.config.hidden_size)

            # Encode segments with segment-wise transformer
            seg_encoder_outputs = self.seg_encoder(encoder_outputs)

            # Collect document representation
            outputs, _ = torch.max(seg_encoder_outputs, 1)

            return SimpleOutput(last_hidden_state=outputs, hidden_states=outputs)
        else:
            # Gather CLS outputs per segment --> (4, 64, 768)
            encoder_outputs = encoder_outputs[:, :, 0]

            # Infer real segments, i.e., mask paddings
            seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
            # Infer and collect segment positional embeddings
            seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
            # Add segment positional embeddings to segment inputs
            encoder_outputs += self.seg_pos_embeddings(seg_positions)

            # Encode segments with segment-wise transformer
            seg_encoder_outputs = self.seg_encoder(encoder_outputs)

            # Collect document representation
            outputs, _ = torch.max(seg_encoder_outputs, 1)

            # Use the dense layer in order to reduce the BERT's output dimension from 768 to 8.
            reduced_embedding = self.first_dense_layer(outputs)

            # Perform the concatenation with the equivalent financial embedding
            concatenated_embedding = torch.cat((reduced_embedding, financial_features), dim=1)

            reduced_concatenated_embedding = self.second_dense_layer(concatenated_embedding)

            return SimpleOutput(last_hidden_state=reduced_concatenated_embedding,
                                hidden_states=reduced_concatenated_embedding)
