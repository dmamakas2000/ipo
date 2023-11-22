#!/usr/bin/env python
# coding=utf-8

"""
File: ipo_longformer.py
Author: Dimitrios Mamakas (Athens University of Economics and Business)
Date: November 22, 2023
Description: Implementation of the following Longformer-based variants.
                • ipo-longformer-extra-global-tx/txff-cls-8192
                • ipo-longformer-extra-global-tx/txff-cls-20480
                • ipo-longformer-tx/txff-cls-8192
                • ipo-longformer-tx/txff-cls-8192


License:
This code is provided under the MIT License.
You are free to copy, modify, and distribute the code.
If you use this code in your research, please include a reference to the original study (please visit the home page).
"""

import os
import sys
import glob
import copy
import torch
import random
import shutil
import logging
import datasets
import numpy as np
import transformers
from typing import Optional
from datasets import load_dataset
from dataclasses import dataclass, field
from trainer.trainer import BinaryTrainer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint
from functions.functions import segment_string, new_compute_metrics
from models.financial_features_longformer import LongformerFinancialModelForSequenceClassification
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LongformerForMaskedLM,
    LongformerForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=8192,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_segments: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum number of segments (paragraphs) to be considered. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seg_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum segment (paragraph) length to be considered. Segments longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory to retrieve the training dataset from."
        }
    )
    eval_dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory to retrieve the evaluation dataset from."
        }
    )
    test_dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory to retrieve the test dataset from."
        }
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    concatenate_financial_features: bool = field(
        default=False,
        metadata={
            "help": "Whether to concatenate financial features (output embedding layer) in order to "
                    "perform the prediction or not."
        }
    )
    reduction_features: int = field(
        default=8,
        metadata={
            "help": "The dimension of the output embedding representation in case it is asked."
        },
    )
    threshold: float = field(
        default=0.5,
        metadata={
            "help": "The threshold to classify texts with."
        }
    )


def extend_longformer(config, model_args, max_pos, attention_window, model_type=AutoModelForMaskedLM):
    """
    Implementation of models:   longformer-extended/extra-global-tx/txff-X, where X represents the model's size.
    """
    # Load standard longformer and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = model_type.from_pretrained(
        'allenai/longformer-base-4096',
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Change tokenizer configuration parameters
    tokenizer.model_max_length = max_pos

    # Init new position embeddings
    max_pos += 2
    current_max_pos, embed_size = model.longformer.embeddings.position_embeddings.weight.shape
    new_pos_embed = model.longformer.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # Copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    new_pos_embed[:k] = model.longformer.embeddings.position_embeddings.weight[:k]
    while k < max_pos - 1:
        if k + step >= max_pos:
            new_pos_embed[k:] = model.longformer.embeddings.position_embeddings.weight[2:(max_pos + 2 - k)]
        else:
            new_pos_embed[k:(k + step)] = model.longformer.embeddings.position_embeddings.weight[2:]
        k += step
    model.longformer.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # Change model configuration parameters
    model.config.attention_window = [attention_window] * model.config.num_hidden_layers
    model.config.max_position_embeddings = max_pos - 2
    model.longformer.embeddings.position_embeddings.weight.data = new_pos_embed
    model.longformer.embeddings.position_embeddings.num_embeddings = max_pos
    return model, tokenizer


def longformerize_bert(config, model_args, max_pos, attention_window, model_type=AutoModelForMaskedLM):
    # Load standard bert and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/sec-bert-base')
    model = model_type.from_pretrained(
        'nlpaueb/sec-bert-base',
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Change tokenizer configuration parameters
    tokenizer.model_max_length = max_pos

    # Init new position embeddings
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # Copy position embeddings over and over to initialize the new position embeddings
    k = 1
    step = current_max_pos
    new_pos_embed[:k] = model.bert.embeddings.position_embeddings.weight[:k]
    while k < max_pos - 1:
        if k + step >= max_pos:
            new_pos_embed[k:] = model.bert.embeddings.position_embeddings.weight[1:(max_pos + 1 - k)]
        else:
            new_pos_embed[k:(k + step - 1)] = model.bert.embeddings.position_embeddings.weight[1:]
        k += step
    model.bert.embeddings.position_embeddings.weight.data = new_pos_embed
    model.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # Add global attention and init with local attention weights
    for i in range(len(model.bert.encoder.layer)):
        model.bert.encoder.layer[i].attention.self.query_global = copy.deepcopy(
            model.bert.encoder.layer[i].attention.self.query)
        model.bert.encoder.layer[i].attention.self.key_global = copy.deepcopy(
            model.bert.encoder.layer[i].attention.self.key)
        model.bert.encoder.layer[i].attention.self.value_global = copy.deepcopy(
            model.bert.encoder.layer[i].attention.self.value)

    # Transfer bert weights to longformer model
    model.config.attention_window = [attention_window] * model.config.num_hidden_layers
    model.config.max_position_embeddings = max_pos
    lfm = None
    if model_type == AutoModelForMaskedLM:
        lfm = LongformerForMaskedLM(model.config)
        lfm.longformer.load_state_dict(model.bert.state_dict())
        lfm.lm_head.dense.load_state_dict(model.cls.predictions.transform.dense.state_dict())
        lfm.lm_head.layer_norm.load_state_dict(model.cls.predictions.transform.LayerNorm.state_dict())
        lfm.lm_head.decoder.load_state_dict(model.cls.predictions.decoder.state_dict())
        lfm.lm_head.bias = copy.deepcopy(model.cls.predictions.bias)
    elif model_type == AutoModelForSequenceClassification:
        pooler = copy.deepcopy(model.bert.pooler.dense)
        delattr(model.bert, 'pooler')
        if model_args.concatenate_financial_features:
            # Financial Longformer
            lfm = LongformerFinancialModelForSequenceClassification(model.config)
            lfm.longformer.load_state_dict(model.bert.state_dict())
            lfm.classifier.dense.load_state_dict(pooler.state_dict())
        else:
            lfm = LongformerForSequenceClassification(model.config)
            lfm.longformer.load_state_dict(model.bert.state_dict())
            lfm.classifier.dense.load_state_dict(pooler.state_dict())

    # Change model configuration parameters
    lfm.longformer.embeddings.position_embeddings.padding_idx = -1
    lfm.longformer.embeddings.padding_idx = -1
    lfm.longformer.embeddings.position_embeddings.num_embeddings = max_pos
    return lfm, tokenizer


def main():
    """
        Main method.
    """
    # Arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load the dataset splits
    if training_args.do_train:
        train_dataset = load_dataset("json", data_files=data_args.train_dataset_dir, split="train",
                                     cache_dir=model_args.cache_dir)

    if training_args.do_eval:
        eval_dataset = load_dataset("json", data_files=data_args.eval_dataset_dir, split="train",
                                    cache_dir=model_args.cache_dir)

    if training_args.do_predict:
        predict_dataset = load_dataset("json", data_files=data_args.test_dataset_dir, split="train",
                                       cache_dir=model_args.cache_dir)

    # Labels
    label_list = list(range(1))
    num_labels = len(label_list)

    # Longformer Config
    config = AutoConfig.from_pretrained(
        'allenai/longformer-base-4096',
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.concatenate_financial_features:
        config.reduction_features = model_args.reduction_features

    # BERT Config
    sec_bert_config = AutoConfig.from_pretrained(
        'nlpaueb/sec-bert-base',
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.concatenate_financial_features:
        sec_bert_config.reduction_features = model_args.reduction_features

    # Load pretrained model and tokenizer
    if model_args.model_name_or_path == 'longformer-standard':
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.model_max_length = 4096
        model = AutoModelForSequenceClassification.from_pretrained(
            'allenai/longformer-base-4096',
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path in ['longformer-extended', 'longformer-extra-global']:
        model, tokenizer = extend_longformer(config=config, model_args=model_args, max_pos=data_args.max_seq_length, attention_window=data_args.max_seg_length,
                                             model_type=AutoModelForSequenceClassification)

        assert model.config.max_position_embeddings == data_args.max_seq_length
        assert model.longformer.embeddings.position_embeddings.weight.shape[0] == data_args.max_seq_length + 2
        assert tokenizer.model_max_length == data_args.max_seq_length
    elif model_args.model_name_or_path in ['ipo-longformer', 'ipo-longformer-extra-global']:
        # Use CLS embedding to perform the classification
        model, tokenizer = longformerize_bert(config=sec_bert_config, model_args=model_args, max_pos=data_args.max_seq_length,
                                              attention_window=data_args.max_seg_length,
                                              model_type=AutoModelForSequenceClassification)

        assert model.config.max_position_embeddings == data_args.max_seq_length
        assert model.longformer.embeddings.position_embeddings.weight.shape[0] == data_args.max_seq_length
        # tokenized = tokenizer(' a ' * 6032, return_tensors='pt')
        tokenized = tokenizer(' a ' * 6142, return_tensors='pt')
        assert model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    def preprocess_function(examples):
        """
            Preprocesses the examples of a specific batch.
        """
        # Tokenize the texts
        cases = []
        max_position_embeddings = tokenizer.model_max_length
        for text in examples['text']:
            # Split each text into segments of fixed length, and only keep the 64 of them
            text_segments = segment_string(text, data_args.max_seg_length)[:data_args.max_segments]

            if model_args.model_name_or_path in ['ipo-longformer-extra-global', 'longformer-extra-global']:
                # If we set the extra-global setting, append a [SEP] token at the end of each paragraph
                cases.append(f' {tokenizer.sep_token} '.join(text_segments))
            else:
                cases.append(f" {' '} ".join(text_segments))

        batch = tokenizer(cases, padding=padding, max_length=max_position_embeddings, truncation=True)
        global_attention_mask = np.zeros((len(cases), max_position_embeddings), dtype=np.int32)
        # global attention on cls token
        global_attention_mask[:, 0] = 1
        batch['global_attention_mask'] = list(global_attention_mask)
        # batch["labels"] = [[1 if labels == label else 0 for label in label_list] for labels in examples["class"]]
        batch["labels"] = [[0 if labels == label else 1 for label in label_list] for labels in examples["class"]]
        if model_args.concatenate_financial_features:
            batch['financial_features'] = examples['financial']
        return batch

    # If training, apply the preprocessing and log a few random samples
    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # If evaluating, apply the preprocessing and log a few random samples
    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # If predicting, apply the preprocessing and log a few random samples
    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = BinaryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=lambda p: new_compute_metrics(p, model_args.threshold),
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save the results
        output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                try:
                    for index, pred_list in enumerate(predictions[0]):
                        pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                        writer.write(f"{index}\t{pred_line}\n")
                except:
                    try:
                        for index, pred_list in enumerate(predictions):
                            pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                            writer.write(f"{index}\t{pred_line}\n")
                    except:
                        pass

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
