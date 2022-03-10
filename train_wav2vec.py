""" Train XLSR model on the training set
TODO
- test if augmentations work correctly
- experiment with parameters (lower learning rate)
- test if predictions are made correctly in the eval function (np.argmax)
- add support for resuming training from checkpoint
"""
import os
import sys
import logging
import random

from typing import Union, Optional

import numpy as np
import torch
import torchaudio
import wandb

import torch_audiomentations

import dataclasses
from dataclasses import dataclass, field
from datasets import load_dataset

from pydantic import validate_arguments

import transformers
from transformers import (
    HfArgumentParser,
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    #    Wav2Vec2ForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from functools import partial

# implement option to resume from checkpoint

from src.data_collator import (
    DataCollatorCTCWithInputPadding,
    DataCollatorCTCWithPaddingKlaam,
)
from src.trainer import CTCTrainer
from src.processor import CustomWav2Vec2Processor
from src.model import Wav2Vec2ForSequenceClassification
from src.make_windows import stack_frames


# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


logger = logging.getLogger(__name__)

@validate_arguments
@dataclass
class IOArguments:
    model_name: str = field(
        default="facebook/wav2vec2-large-xlsr-53",
        metadata={"help": "path to pretrained model or identifer from huggingface hub"},
    )
    train: str = field(
        default=os.path.join("data", "splits", "train.csv"),
        metadata={"help": "path to train data csv"},
    )
    validation: str = field(
        default=os.path.join("data", "splits", "val.csv"),
        metadata={"help": "path to validation data csv"},
    )
    input_col: str = field(
        default="file", metadata={"help": "name of column in csv with input files"}
    )
    label_col: str = field(
        default="Diagnosis", metadata={"help": "name of column in csv with labels"}
    )
    use_windowing: bool = field(
        default=True, metadata={"help": "segment files into windows"}
    )
    window_length: int = field(default=5, metadata={"help": "window length in seconds"})
    stride_length: Union[int, float] = field(
        default=1.0, metadata={"help": "stride length in seconds"}
    )
    augmentations: str = field(
        default="",
        metadata={
            "help": "path to config yml for torch-audiomentations. Empty if no augmentations are desired"
        },
    )
    max_training_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

@validate_arguments
@dataclass
class ModelArguments:
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout rate for attention layers"}
    )
    hidden_dropout: float = field(
        default=0.1, metadata={"help": "dropout rate for hidden layers"}
    )
    final_dropout: float = field(
        default=0.1, metadata={"help": "dropout rate for final layer"}
    )
    feat_proj_dropout: float = field(
        default=0.2, metadata={"help": "dropout rate for feature projection"}
    )
    mask_time_prob: float = field(
        default=0.05, metadata={"help": "probability of masking time dimension"}
    )
    layerdrop: float = field(default=0.1, metadata={"help": "layer dropout rate"})
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "enable gradient checkpointing"}
    )
    ctc_loss_reduction: str = field(
        default="sum", metadata={"help": "reduction for ctc loss"}
    )
    freeze_encoder: bool = field(
        default=True, metadata={"help": "freeze encoder weights during training"}
    )
    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "freeze entire base model weights during training"},
    )


# Preprocessing functions
def speech_file_to_array(path):
    "resample audio to match what the model expects (16000 khz)"
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def stack_speech_file_to_array(path):
    """Loads and resamples audio to target sampling rate and converts the
    audio into windows of specified length and stride"""
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech_array = resampler(speech_array)

    windowed_arrays = stack_frames(
        speech_array.squeeze(),
        sampling_rate=target_sampling_rate,
        frame_length=io_args.window_length,
        frame_stride=io_args.stride_length,
    )
    return windowed_arrays


def preprocess_stacked_speech_files(batch):
    """Process batch of audio files into windows of io_args.window_length with io_args.stride_length
    and return input values as well as metadata for the batch"""
    speech_list = [
        stack_speech_file_to_array(path) for path in batch[io_args.input_col]
    ]
    labels = [label_to_id(label, label_list) for label in batch[io_args.label_col]]
    n_windows = [len(window) for window in speech_list]

    processed_list = [
        processor(speech_window, sampling_rate=target_sampling_rate)
        for speech_window in speech_list
    ]

    # make new larger dictionary that contains the flattened values
    # labels = label as idx
    out = {"input_values": [], "attention_mask": [], "labels": []}
    # save metadata from other columns
    for meta_key in batch.keys():
        out[meta_key] = []
    # looping through list of processed stacked speech arrays
    for i, processed_speech in enumerate(processed_list):
        # un-nesting the stacked time windows
        for key, value in processed_speech.items():
            # values are indented in a list, need to index 0 to get them out
            out[key].extend(value[0])
        # making sure each window has the right label
        out["labels"] += [labels[i]] * n_windows[i]
        # adding metadata to be able to reidentify files
        for meta_key, meta_value in batch.items():
            out[meta_key] += [meta_value[i]] * n_windows[i]

    return out


def label_to_id(label, label_list):
    "map label to id int"
    return label_list.index(label)


def preprocess(batch):
    "preprocess hf dataset/load data"
    speech_list = [speech_file_to_array(path) for path in batch[io_args.input_col]]
    labels = [label_to_id(label, label_list) for label in batch[io_args.label_col]]

    out = processor(speech_list, sampling_rate=target_sampling_rate)
    out["labels"] = list(labels)
    return out


# which metrics to compute for evaluation
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# def compute_metrics(pred):
#     labels = pred.label_ids.argmax(-1)
#     preds = pred.predictions.argmax(-1)
#     acc = accuracy_score(labels, preds)
#     wandb.log(
#         {"conf_mat" : wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=label_list)}
#     )
#     wandb.log(
#         {"precision_recall" : wandb.plot.pr_curve(y_true=labels, preds=preds, class_names=label_list)}
#     )
#     return {"accuracy": acc}


if __name__ == "__main__":
    # see possible arguments by passing --help to the script
    parser = HfArgumentParser((IOArguments, ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # loading parameters from json config file if supplied
        io_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        io_args, model_args, training_args = parser.parse_args_into_dataclasses()

    ##### setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("IO arguments %s", io_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    ################### LOAD DATASETS
    #######
    ####
    # load datasets
    data_files = {
        "train": io_args.train,
        "validation": io_args.validation,
    }

    print("[INFO] Loading dataset...")
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")
    train = dataset["train"]
    val = dataset["validation"]

    # Optionally train on fewer samples for debugging
    if io_args.max_training_samples is not None:
        n_train_samples = len(train)
        assert io_args.max_training_samples < n_train_samples
        train_indices = random.sample(
            range(1, n_train_samples), io_args.max_training_samples
        )
        train = train.select(train_indices)
        train = train.flatten_indices()
    if io_args.max_validation_samples is not None:
        n_val_samples = len(val)
        assert io_args.max_validation_samples < n_val_samples
        val_indices = random.sample(
            range(1, n_val_samples), io_args.max_validation_samples
        )
        val = val.select(val_indices)
        val = val.flatten_indices()
    # get labels and num labels
    label_list = train.unique(io_args.label_col)
    # sorting for determinism
    label_list.sort()
    num_labels = len(label_list)

    # train = train.select([0])

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(io_args.model_name)
    processor = CustomWav2Vec2Processor(feature_extractor=feature_extractor)
    # need this parameter for preprocessing to resample audio to correct sampling rate
    target_sampling_rate = processor.feature_extractor.sampling_rate

    # preprocess datasets
    print("[INFO] Preprocessing dataset...")
    if io_args.use_windowing:
        print(
            f"[INFO] Using windows of size {io_args.window_length} and stride {io_args.stride_length}"
        )
        train = train.map(
            preprocess_stacked_speech_files,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        val = val.map(
            preprocess_stacked_speech_files,
            batched=True,
            remove_columns=dataset["validation"].column_names,
        )
    else:
        train = train.map(preprocess, batched=True)
        val = val.map(preprocess, batched=True)

    # shuffle rows of training set (necesarry?)
    train = train.shuffle(seed=42)

    ################### LOAD MODEL
    #######
    ####

    # loading model config
    config = AutoConfig.from_pretrained(
        io_args.model_name,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        **dataclasses.asdict(model_args),
    )

    # load model (with a simple linear projection (input 1024 -> 256 units) and a classification layer on top)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        io_args.model_name, config=config
    )

    # instantiate a data collator that takes care of correctly padding and optionally augmenting the input data
    if io_args.augmentations:
        augmenter = torch_audiomentations.utils.config.from_yaml(io_args.augmentations)
        augment_fn = partial(augmenter, sample_rate=target_sampling_rate)
        data_collator = DataCollatorCTCWithInputPadding(
            processor=processor, padding=True, augmentation_fn=augment_fn
        )
    else:
        data_collator = DataCollatorCTCWithInputPadding(
            processor=processor, padding=True
        )

    if model_args.freeze_encoder and not model_args.freeze_base_model:
        model.freeze_feature_extractor()
        print("[INFO] Freezing encoder...")
    if model_args.freeze_base_model:
        model.freeze_base_model()
        print("[INFO] Freezing entire base model...")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=processor.feature_extractor,
    )

    # Train!
    print("[INFO] Starting training...")
    train_result = trainer.train()
    print("[INFO] Training finished!")
    trainer.save_model()
    # save the feature_extractor and the tokenizer
    if is_main_process(training_args.local_rank):
        feature_extractor.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["n_train_samples"] = len(train)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)  # save metrics to file
    trainer.save_state()

    # evaluate model on test set
    trainer.evaluate()
    metrics["eval_samples"] = len(val)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
