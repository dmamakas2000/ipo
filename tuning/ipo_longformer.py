#!/usr/bin/env python
# coding=utf-8

"""
File: ipo_longformer.py
Author: Dimitrios Mamakas (Athens University of Economics and Business)
Date: November 23, 2023
Description: Implementation of the tuning strategy followed for Longformer-based variants.


License:
This code is provided under the MIT License.
You are free to copy, modify, and distribute the code.
If you use this code in your research, please include a reference to the original study (please visit the home page).
"""

import os
import json
import optuna
import logging
import numpy as np
from typing import Optional
from datasets import load_dataset
from dataclasses import dataclass, field
from trainer.trainer import BinaryTrainer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from experiments.ipo_longformer import extend_longformer, longformerize_bert
from functions.functions import segment_string, new_compute_metrics, clean_checkpoints, detect_last_checkpoint, \
    setup_logging
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
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


def objective(trial):
    """
    Performs a trial.
    """

    # Arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Define hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    threshold = trial.suggest_float('threshold', 0.45, 0.65)

    if data_args.max_seq_length <= 8192:
        batch_size = trial.suggest_int('batch_size', 1, 3)
    else:
        batch_size = trial.suggest_int('batch_size', 1, 1)

    # Modify the training arguments as needed
    training_args.learning_rate = learning_rate
    training_args.per_device_train_batch_size = batch_size
    training_args.per_device_eval_batch_size = batch_size
    training_args.num_train_epochs = 5

    # Tune the output embedding size if specified
    if model_args.concatenate_financial_features:
        reduction_features = trial.suggest_int('reduction_features', 8, 16)
        training_args.reduction_features = reduction_features

    # Ensure that the output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Save the current trial number into a .txt file (used to monitor the progress by watching the console)
    with open(training_args.output_dir + '/current_trial.txt', 'w') as file:
        file.write(f"Current Trial is: {trial.number}:")

    # Setup logging
    setup_logging(logger, training_args)

    # Detecting last checkpoint.
    last_checkpoint = detect_last_checkpoint(logger, training_args)

    # Load the dataset splits
    train_dataset = load_dataset("json", data_files=data_args.train_dataset_dir, split="train", cache_dir=model_args.cache_dir)
    eval_dataset = load_dataset("json", data_files=data_args.eval_dataset_dir, split="train", cache_dir=model_args.cache_dir)

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

    # BERT Config
    sec_bert_config = AutoConfig.from_pretrained(
        'nlpaueb/sec-bert-base',
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Concatenate financial features if specified
    if model_args.concatenate_financial_features:
        config.reduction_features = model_args.reduction_features
        sec_bert_config.reduction_features = model_args.reduction_features

    # Load pretrained model and tokenizer
    model = None
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
        batch["labels"] = [[0 if labels == label else 1 for label in label_list] for labels in examples["class"]]
        if model_args.concatenate_financial_features:
            batch['financial_features'] = examples['financial']
        return batch

    # If training, apply the preprocessing and log a few random samples
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
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
        compute_metrics=lambda p: new_compute_metrics(p, threshold),
        tokenizer=tokenizer,
        data_collator=data_collator,
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

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    # Clean up checkpoints
    clean_checkpoints(training_args.output_dir)

    # Evaluate the model using validation metrics
    val_loss = metrics['eval_loss']
    return val_loss


def main():
    # Again, load the arguments (used to save the optimal parameters returned as a json file)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create a new study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)
    best_params = study.best_params

    # Create a dictionary containing the optimal parameters
    if model_args.concatenate_financial_features:
        optimal_parameters = {
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size'],
            'reduction_features': best_params['reduction_features'],
            'threshold': best_params['threshold']
        }
    else:
        optimal_parameters = {
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size'],
            'threshold': best_params['threshold']
        }

    # Save optimal parameters
    with open(training_args.output_dir + '/optimal_parameters.json', 'w') as json_file:
        json.dump(optimal_parameters, json_file)
    json_file.close()


if __name__ == "__main__":
    main()
