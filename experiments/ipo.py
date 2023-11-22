#!/usr/bin/env python
# coding=utf-8

import os
import sys
import glob
import shutil
import random
import logging
import datasets
import transformers
from typing import Optional
from datasets import load_dataset
from dataclasses import dataclass, field

from models.financial_features_hierarchical_bert import FinancialHierarchicalBert, \
    HierarchicalBertFinancialModelForSequenceClassification
from trainer.trainer import BinaryTrainer
from transformers.utils import check_min_version
from models.hierarchical_bert import HierarchicalBert
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint
from trainer.financial_features_trainer import FinancialTrainer
from functions.functions import segment_string, new_compute_metrics
from models.max_pooled_bert import BertMaxPooledForSequenceClassification
from models.financial_features_bert import BertFinancialModelForSequenceClassification
from models.max_pooled_financial_features_bert import FinancialBertMaxPooledForSequenceClassification
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# Will error if the minimal version of Transformers is not installed
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
        default=512,
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
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
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
        }
    )
    max_pooled: bool = field(
        default=False,
        metadata={
            "help": "Whether to use a max-pooled embedding as an input into the classification head."
                    "If set to False, the CLS embedding will be used to perform the classification."
        }
    )
    hierarchical: bool = field(
        default=False, metadata={"help": "Whether to use a hierarchical variant or not."}
    )
    concatenate_financial_features: bool = field(
        default=False, metadata={"help": "Whether to concatenate the financial features among with the textual, or not."}
    )
    reduction_features: int = field(
        default=8,
        metadata={
            "help": "The number of output BERT features to keep in case it is asked."
        },
    )
    multiple_dense_layers: bool = field(
        default=True,
        metadata={
            "help": "Whether to use a second dense layer on top of the first one (if selected), or not."
        },
    )
    threshold: float = field(
        default=0.5,
        metadata={
            "help": "The threshold to classify texts with."
        }
    )



def main():
    """
        Main method.
    """
    # Arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Fix boolean parameter
    if model_args.do_lower_case == 'False' or not model_args.do_lower_case:
        model_args.do_lower_case = False
        'Tokenizer do_lower_case False'
    else:
        model_args.do_lower_case = True

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

    # Config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Set additional parameters to control the flow of the experiments
    config.reduction_features = model_args.reduction_features
    config.multiple_dense_layers = model_args.multiple_dense_layers

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Check all possible combinations for choosing model and setting
    if not model_args.hierarchical:
        if model_args.max_pooled:
            if model_args.concatenate_financial_features:
                """
                Scenario 1: BERT (max-pooled) using financial embeddings.
                """
                model = FinancialBertMaxPooledForSequenceClassification.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
            else:
                """
                Scenario 2: BERT (max-pooled).
                """
                model = BertMaxPooledForSequenceClassification.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
        else:
            if model_args.concatenate_financial_features:
                """
                Scenario 3: BERT (cls-pooled) using financial embeddings.
                """
                model = BertFinancialModelForSequenceClassification.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
            else:
                """
                Scenario 4: BERT (cls-pooled).
                """
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )

    if model_args.hierarchical:
        if model_args.concatenate_financial_features:
            """
            Scenario 5: Hierarchical-BERT using financial embeddings.
            """
            model = HierarchicalBertFinancialModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            segment_encoder = model.bert
            model_encoder = FinancialHierarchicalBert(config=config,
                                                      encoder=segment_encoder,
                                                      max_segments=data_args.max_segments,
                                                      max_segment_length=data_args.max_seg_length,
                                                      max_pooled=model_args.max_pooled)
            model.bert = model_encoder
        else:
            """
            Scenario 6: Hierarchical-BERT.
            """
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            segment_encoder = model.bert
            model_encoder = HierarchicalBert(encoder=segment_encoder,
                                             max_segments=data_args.max_segments,
                                             max_segment_length=data_args.max_seg_length,
                                             max_pooled=model_args.max_pooled)
            model.bert = model_encoder

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    def preprocess_function(examples):
        """
            Preprocesses the examples of a specific batch.
        """
        if model_args.hierarchical:
            case_template = [[0] * data_args.max_seg_length]
            batch = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
            for doc in examples['text']:
                doc = segment_string(doc, data_args.max_seg_length)
                doc_encodings = tokenizer(doc[:data_args.max_segments], padding=padding,
                                          max_length=data_args.max_seg_length, truncation=True)
                batch['input_ids'].append(doc_encodings['input_ids'] + case_template * (
                        data_args.max_segments - len(doc_encodings['input_ids'])))
                batch['attention_mask'].append(doc_encodings['attention_mask'] + case_template * (
                        data_args.max_segments - len(doc_encodings['attention_mask'])))
                batch['token_type_ids'].append(doc_encodings['token_type_ids'] + case_template * (
                        data_args.max_segments - len(doc_encodings['token_type_ids'])))
        else:
            # Tokenize the texts
            batch = tokenizer(
                examples["text"],
                padding=padding,
                max_length=data_args.max_seq_length,
                truncation=True,
            )
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

    # Trainer
    if model_args.concatenate_financial_features:
        trainer = FinancialTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=lambda p: new_compute_metrics(p, model_args.threshold),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = BinaryTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=lambda p: new_compute_metrics(p, model_args.threshold),
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
        # trainer.save_model()
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
