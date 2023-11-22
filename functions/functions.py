"""
File: ipo_longformer.py
Author: Dimitrios Mamakas (Athens University of Economics and Business)
Date: November 22, 2023
Description: Implementation of the essential functions used across this project.


License:
This code is provided under the MIT License.
You are free to copy, modify, and distribute the code.
If you use this code in your research, please include a reference to the original study (please visit the home page).
"""

import os
import sys
import glob
import shutil
import logging
import datasets
import transformers
import numpy as np
from transformers import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import precision_score, recall_score, auc


def calculate_f1_score_per_class(precision, recall):
    """
    Calculates the f1-score for each distinct class
    """

    if precision + recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)


def calculate_auc_scores(probs, true):
    """
    Calculates the AUC scores given some data, their true labels, and the estimator to be used.
    """

    threshold = 0.0
    y_predictions = []

    for i in range(0, 12):
        p = []
        for prob in probs:
            if prob >= threshold:
                p.append(1)
            else:
                p.append(0)
        y_predictions.append(p)
        threshold += 0.1

    precision_0 = []
    recall_0 = []
    precision_1 = []
    recall_1 = []
    for i in range(0, 12):
        pr = precision_score(true, y_predictions[i], zero_division=0, average=None)
        rec = recall_score(true, y_predictions[i], zero_division=0, average=None)

        pr_0 = pr[0]
        pr_1 = pr[1]
        rec_0 = rec[0]
        rec_1 = rec[1]

        precision_0.append(pr_0)
        recall_0.append(rec_0)
        precision_1.append(pr_1)
        recall_1.append(rec_1)

    auc_precision_recall_0 = auc(np.array(recall_0), np.array(precision_0))
    auc_precision_recall_1 = auc(np.array(recall_1), np.array(precision_1))
    return auc_precision_recall_0, auc_precision_recall_1


def sig(x):
    """
    Implements the sigmoid function for given x.
    """

    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Implements the softmax function for given x.
    """

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def new_compute_metrics(p: EvalPrediction, threshold_):
    """
    Computes the model's metrics.
    """

    # Set a threshold to work with
    threshold = threshold_

    # True values
    targets_ = p.label_ids.astype(np.int32)
    targets = []
    for x in targets_:
        if x[0] == 1:
            targets.append(1)
        else:
            targets.append(0)

    # Convert targets to numpy array
    targets = np.array(targets)

    # Predictions
    predictions = []
    probabilities = []
    if isinstance(p.predictions, tuple):
        for x in p.predictions[0]:
            p = sig(x)
            probabilities.append(p)
            if p >= threshold:
                predictions.append(1)
            else:
                predictions.append(0)
    else:
        for x in p.predictions:
            p = sig(x)
            probabilities.append(p)
            if p >= threshold:
                predictions.append(1)
            else:
                predictions.append(0)

    # Convert predictions to numpy array
    predictions = np.array(predictions)

    # Compute metric scores
    precision = precision_score(targets, predictions, zero_division=0, average=None)
    recall = recall_score(targets, predictions, zero_division=0, average=None)

    precision_0 = precision[0]
    precision_1 = precision[1]
    recall_0 = recall[0]
    recall_1 = recall[1]

    f1_0 = calculate_f1_score_per_class(precision_0, recall_0)
    f1_1 = calculate_f1_score_per_class(precision_1, recall_1)

    auc_0, auc_1 = calculate_auc_scores(probabilities, targets)

    # Macro averaging
    macro_avg_precision = (precision_0 + precision_1) / 2
    macro_avg_recall = (recall_0 + recall_1) / 2
    macro_avg_f1 = (f1_0 + f1_1) / 2
    macro_avg_pr_auc = (auc_0 + auc_1) / 2
    metrics = {
        'precision-class-0': precision_0,
        'precision-class-1': precision_1,
        'recall-class-0': recall_0,
        'recall-class-1': recall_1,
        'pr-auc-class-0': auc_0,
        'pr-auc-class-1': auc_1,
        'f1-class-0': f1_0,
        'f1-class-1': f1_1,
        'macro-avg-precision': macro_avg_precision,
        'macro-avg-recall': macro_avg_recall,
        'macro-avg-auc': macro_avg_pr_auc,
        'macro-avg-f1': macro_avg_f1
    }
    return metrics


def segment_string(text, segment_length):
    """
    Splits the text into segments of fixed length.
    """
    segments = []
    words = text.split()
    # Split the words into segments
    for i in range(0, len(words), segment_length):
        segment = ' '.join(words[i:i + segment_length])
        segments.append(segment)
    return segments


def clean_checkpoints(output_dir):
    """
    Cleans up checkpoints.
    """
    checkpoints = [filepath for filepath in glob.glob(f'{output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)

    runs = [filepath for filepath in glob.glob(f'{output_dir}/*/') if '/runs' in filepath]
    for run in runs:
        shutil.rmtree(run)


def setup_logging(logger, training_args):
    """
    Sets up logging for experiments.
    """
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


def detect_last_checkpoint(logger, training_args):
    """
    Detects and returns the last checkpoint.
    """
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
    return last_checkpoint
