"""Train wav2vec for autism classification (on both stories and triangles task - probably do one for each)
TODO
- experiment with parameters (lower learning rate)
- test if predictions are made correctly in the eval function (np.argmax)
"""

import numpy as np
import torch
import torchaudio
import os
from datasets import load_dataset
import wandb
# from model import Wav2Vec2ForSpeechClassification

from src.data_collator import DataCollatorCTCWithInputPadding, DataCollatorCTCWithPaddingKlaam
from src.trainer import CTCTrainer
from src.processor import CustomWav2Vec2Processor
from src.model import Wav2Vec2ForSequenceClassification
from src.make_windows import stack_frames
from dataclasses import dataclass, field

from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
#    Wav2Vec2ForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# set constants
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
OUTPUT_DIR = os.path.join("model", "xlsr_autism_stories")
TRAIN = os.path.join("data", "splits", "stories_train_data_gender_False.csv")
VALIDATION = os.path.join("data", "splits", "stories_test_data_gender_False.csv")
RUN_NAME = "frozen_encoder_0.1_final_drop_stories"
RUN_NAME = "debug"

FREEZE_ENCODER = True
FREEZE_BASE_MODEL = False
# specify input/label columns
INPUT_COL = "file"
LABEL_COL = "Diagnosis"
# training params
EPOCHS = 100
LEARNING_RATE = 3e-3  # 3e-3
BATCH_SIZE = 16

# windowing parameters
USE_WINDOWING = True
WINDOW_LENGTH = 5 # window length in seconds
STRIDE_LENGTH = 1 # stride length in seconds

# model params              # default
ATTENTION_DROPOUT = 0.1     # 0.1
HIDDEN_DROPOUT = 0.1        # 0.1
FINAL_DROPOUT = 0.1 
FEAT_PROJ_DROPOUT=0.2       # 0.1
MASK_TIME_PROB=0.05         # 0.075
LAYERDROP = 0.1             # 0.1
GRADIENT_CHECKPOINTING=True # False
CTC_LOSS_REDUCTION="sum"   # "sum"   - try "mean"

params = {"attention_dropout" : ATTENTION_DROPOUT,
          "hidden_dropout" : HIDDEN_DROPOUT,
          "feat_proj_dropout" : FEAT_PROJ_DROPOUT,
          "mask_time_prob" : MASK_TIME_PROB,
          "layerdrop" : LAYERDROP,
          "gradient_checkpointing" : GRADIENT_CHECKPOINTING,
          "ctc_loss_reduction" : CTC_LOSS_REDUCTION,
          "final_dropout" : FINAL_DROPOUT}

# Preprocessing functions
def speech_file_to_array(path):
    "resample audio to match what the model expects (16000 khz)"
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def stack_speech_file_to_array(path):
    speech_array, sampling_rate = torchaudio.load(path)
    windowed_arrays = stack_frames(speech_array.squeeze(), sampling_rate=sampling_rate,
                        frame_length=WINDOW_LENGTH, frame_stride=STRIDE_LENGTH)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    windowed_arrays = [resampler(window).squeeze() for window in windowed_arrays]
    return windowed_arrays


def flatten(t):
    return [item for sublist in t for item in sublist]


def preprocess_stacked_speech_files(batch):
    speech_list = [stack_speech_file_to_array(path) for path in batch[INPUT_COL]]
    labels = [label_to_id(label, label_list) for label in batch[LABEL_COL]]
    n_windows = [len(window) for window in speech_list]


    processed_list = [processor(speech_window, sampling_rate=target_sampling_rate) 
        for speech_window in speech_list]

    # make 'out' contain metadata
    # try using + 

    out = {"input_values" : [], "attention_mask" : [], "labels" : []}
    for meta_key in batch.keys():
        out[meta_key] = []
    # looping through list of processed stacked speech arrays
    for i, processed_speech in enumerate(processed_list):
        # un-nesting the stacked time windows 
        for key, value in processed_speech.items():
            out[key].append(value)
        # making sure each window has the right label
        out["labels"].append([labels[i]] * n_windows[i])
        # adding metadata to be able to reidentify files
        for meta_key, meta_value in batch.items():
            out[meta_key].append([meta_value] * n_windows[i])
    # un-nesting list again
    for key, value in out.items():
        out[key] = flatten(value)

    return out

# path = dataset["train"]["file"][0]
# sp = stack_speech_file_to_array(path)
# sp = speech_file_to_array(path)

# batch = {"file" : [dataset["train"]["file"][1], dataset["train"]["file"][1]],
#          "Diagnosis" : ["ASD", "TD"]}

# goal_out = preprocess(batch)


def label_to_id(label, label_list):
    "map label to id int"
    return label_list.index(label)

def preprocess(batch):
    "preprocess hf dataset/load data"
    speech_list = [speech_file_to_array(path) for path in batch[INPUT_COL]]
    labels = [label_to_id(label, label_list) for label in batch[LABEL_COL]]
    
    out = processor(speech_list, sampling_rate=target_sampling_rate)
    out["labels"] = list(labels)
    return out

# which metrics to compute for evaluationz
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

    ################### LOAD DATASETS
    #######
    ####
    # load datasets
    data_files = {
        "train" : TRAIN,
        "validation" : VALIDATION
    }

    print("[INFO] Loading dataset...")
    dataset = load_dataset("csv", data_files=data_files, delimiter = ",")
    train = dataset["train"]
    val = dataset["validation"]

    # get labels and num labels
    label_list = train.unique(LABEL_COL)
    # sorting for determinism
    label_list.sort()
    num_labels = len(label_list)

    # train = train.select([0])

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    processor = CustomWav2Vec2Processor(feature_extractor=feature_extractor)
    # need this parameter for preprocessing to resample audio to correct sampling rate
    target_sampling_rate = processor.feature_extractor.sampling_rate

    # preprocess datasets
    print("[INFO] Preprocessing dataset...")
    if USE_WINDOWING:
        print(f"[INFO] Using windows of size {WINDOW_LENGTH} and stride {STRIDE_LENGTH}")
        train = train.map(preprocess_stacked_speech_files, batched=True, remove_columns=dataset["train"].column_names)
        val = val.map(preprocess_stacked_speech_files, batched=True, remove_columns=dataset["validation"].column_names)
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
            MODEL_NAME,
            num_labels=num_labels,
            label2id={label: i for i, label in enumerate(label_list)},
            id2label={i: label for i, label in enumerate(label_list)},
            finetuning_task="wav2vec2_clf",
            **params
        )

    # load model (with a simple linear projection (input 1024 -> 256 units) and a binary classification on top)
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", config=config)

    # instantiate a data collator that takes care of correctly padding the input data
    # data_collator = DataCollatorCTCWithInputPadding(processor=processor, padding=True)
    data_collator = DataCollatorCTCWithInputPadding(processor=processor, padding=True)

    if FREEZE_ENCODER and not FREEZE_BASE_MODEL:
        model.freeze_feature_extractor()
        print("Freezing encoder...")
    if FREEZE_BASE_MODEL:
        model.freeze_base_model()
        print("Freezing entire base model...")
 
    # set arguments to Trainer
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        #group_by_length=True, # can speed up training by batching files of similar length to reduce the amount of padding
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2, # try increasing to reduce memory 
        evaluation_strategy="steps",
        num_train_epochs=EPOCHS,
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=LEARNING_RATE, # play with this (also optimizer and learning schedule)
        save_total_limit=2,
        load_best_model_at_end=True,
        run_name = RUN_NAME
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=processor.feature_extractor
    )

    # Train!
    print("[INFO] Starting training...")
    trainer.train()
    trainer.evaluate()
