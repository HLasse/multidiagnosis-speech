#from platform import processor
import numpy as np
import torchaudio
import os
from datasets import load_dataset
# from model import Wav2Vec2ForSpeechClassification
from src.data_collator import DataCollatorCTCWithInputPadding
# from src.trainer import CTCTrainer
from dataclasses import dataclass, field

from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    Trainer)

# set constants
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
# training params
EPOCHS = 100
LEARNING_RATE = 3e-3
# model parameters (play around with these)


# potentially rewrite arguments into this form to be parsable with HfArgumentParser
# @dataclass
# class ModelConfigArguments:
#     """Arguments pertaining to the model config"""
#     attention_dropout: int = field(
#         default=0.1,
#         metadata={
#             "help" : "fill in stuff."
#         }
#     )

# model params              # default
ATTENTION_DROPOUT = 0.1     # 0.1
HIDDEN_DROPOUT = 0.1        # 0.1
FEAT_PROJ_DROPOUT=0.0       # 0.1
MASK_TIME_PROB=0.05         # 0.075
LAYERDROP = 0.1             # 0.1
GRADIENT_CHECKPOINTING=True # False
CTC_LOSS_REDUCTION="mean"   # "sum"



params = {"attention_dropout" : ATTENTION_DROPOUT,
          "hidden_dropout" : HIDDEN_DROPOUT,
          "feat_proj_dropout" : FEAT_PROJ_DROPOUT,
          "mask_time_prob" : MASK_TIME_PROB,
          "layerdrop" : LAYERDROP,
          "gradient_checkpointing" : GRADIENT_CHECKPOINTING,
          "ctc_loss_reduction" : CTC_LOSS_REDUCTION}

# Preprocessing functions
def speech_file_to_array(path):
    "resample audio to match what the model expects (16000 khz)"
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):
    "map label to id int"
    return label_list.index(label)

def preprocess(batch):
    "preprocess hf dataset/load data"
    speech_list = [speech_file_to_array(path) for path in batch[input_col]]
    labels = [label_to_id(label, label_list) for label in batch[label_col]]
    
    out = processor(speech_list, sampling_rate=target_sampling_rate)
    out["labels"] = list(labels)
    return out

# which metrics to compute for evaluation
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


if __name__ == "__main__":

    ################### LOAD DATASETS
    #######
    ####
    # load datasets
    data_files = {
        "train" : os.path.join("preproc_data", "train_data.csv"),
        "validation" : os.path.join("preproc_data", "test_data.csv")
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter = "\t")
    train = dataset["train"]
    val = dataset["validation"]
    # specify input/label columns
    input_col = "file"
    label_col = "label"

    # get labels and num labels
    label_list = train.unique(label_col)
    # sorting for determinism
    label_list.sort()
    num_labels = len(label_list)

    # Load feature extractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    # need this parameter for preprocessing to resample audio to correct sampling rate
    target_sampling_rate = processor.sampling_rate

    # preprocess datasets
    train = train.map(preprocess, batched=True)
    val = val.map(preprocess, batched=True)

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
    data_collator = DataCollatorCTCWithInputPadding(processor=processor, padding=True)

    # freezing the feature extractor (the CNN encoder) of the model - it's already finetuned plenty
    model.freeze_feature_extractor()
    # can potentially also freeze all wav2vec parameters (including the transformer) with
    # model.freeze_base_model()

    # set arguments to Trainer
    training_args = TrainingArguments(
        output_dir = os.path.join("model", "xlsr_gender_recognition"),
        #group_by_length=True, # can speed up training by batching files of similar length to reduce the amount of padding
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=EPOCHS,
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=LEARNING_RATE, # play with this (also optimizer and learning schedule)
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=processor
    )
    # Train!
    trainer.train()
    trainer.evaluate()
