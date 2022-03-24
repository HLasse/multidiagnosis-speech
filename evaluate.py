"""evaluate model performance
TODO
- Evaluate by window and by participant (rewrite to make windows)
- make more efficient by not using apply
    - Make windows first, then use predict function on 
"""

import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchaudio

from typing import Union

import time

from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HfArgumentParser

from datasets import load_dataset

from src.processor import CustomWav2Vec2Processor
from wav2vec_model import Wav2Vec2ForSequenceClassification
from src.make_windows import stack_frames

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import dataclasses
from dataclasses import dataclass, field


@dataclass
class EvalArguments:
    model_path: str = field(default="", metadata={"help": "path to model"})
    data_path: str = field(
        default="data/audio_file_splits/audio_val_split.csv",
        metadata={"help": "path to data to evaluate"},
    )
    input_col: str = field(
        default="file", metadata={"help": "name of column in csv with input file"}
    )
    label_col: str = field(
        default="label", metadata={"help": "name of column in csv with label"}
    )
    save_dir: str = field(
        default="data/eval_results",
        metadata={
            "help": "path to save results. Uses model name as filename if not specified"
        },
    )
    use_windowing: bool = field(
        default=True, metadata={"help": "segment files into windows"}
    )
    window_length: int = field(default=4, metadata={"help": "window length in seconds"})
    stride_length: Union[int, float] = field(
        default=1.0, metadata={"help": "stride length in seconds"}
    )
    performance_by_window: bool = field(
        default=True, metadata={"help": "evaluate performance by window"}
    )
    performance_by_id: bool = field(
        default=True, metadata={"help": "evaluate performance by participant"}
    )


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def stack_speech_file_to_array(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech_array = resampler(speech_array)

    windowed_arrays = stack_frames(
        speech_array.squeeze(),
        sampling_rate=sampling_rate,
        frame_length=eval_args.window_length,
        frame_stride=eval_args.stride_length,
    )
    return windowed_arrays


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = processor(
        speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=0).detach().cpu().numpy()[0]
    pred = config.id2label[np.argmax(scores)]
    confidence = scores[np.argmax(scores)]
    return pred, confidence


def predict_windows(path, sampling_rate, aggregation_fn=lambda x: np.mean(x, axis=0)):
    """Create windows from an input file and output aggregated predictions"""
    speech_windows = stack_speech_file_to_array(path)
    features = processor(
        speech_windows, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )

    # needs to remove first dimension if more than one window
    # squeeze doesn't work if only 1 window (removes two dimensions)
    input_values = features.input_values.to(device).flatten(0, 1)
    attention_mask = features.attention_mask.to(device).flatten(0, 1)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = [F.softmax(logit, dim=0).detach().cpu().numpy() for logit in logits]
    pooled_pred = aggregation_fn(scores)
    pred = config.id2label[np.argmax(pooled_pred)]
    confidence = pooled_pred[np.argmax(pooled_pred)]
    return pred, confidence


def add_predicted_and_confidence(df):
    if eval_args.use_windowing:
        pred, confidence = predict_windows(
            df[eval_args.input_col], target_sampling_rate
        )
    else:
        pred, confidence = predict(df[eval_args.input_col], target_sampling_rate)
    df["pred"] = pred
    df["confidence"] = confidence
    return df


if __name__ == "__main__":

    parser = HfArgumentParser((EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # loading parameters from json config file if supplied
        eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        eval_args = parser.parse_args_into_dataclasses()

    # turned into a tuple with 1 element for some reason, fix it at some point
    eval_args = eval_args[0]
    # setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(eval_args.model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(eval_args.model_path)
    processor = CustomWav2Vec2Processor(feature_extractor=feature_extractor)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    model = Wav2Vec2ForSequenceClassification.from_pretrained(eval_args.model_path).to(
        device
    )



    print("[INFO] Loading dataset...")
    dataset = load_dataset(
        "csv", data_files={"test": eval_args.data_path}, delimiter=","
    )
    test = dataset["test"]
    print(
        f"[INFO] Preprocessing dataset with window size {eval_args.window_length} and stride length {eval_args.stride_length}"
    )
    ### make windows with map
    ### apply model on windows with batched map


    test = pd.read_csv(eval_args.data_path)
    print(f"Evaluating on {eval_args.data_path} containing {len(test)} files")
    # apply predictions
    t0 = time.time()
    test = test.apply(add_predicted_and_confidence, axis=1)
    print(f"Time taken: {time.time() - t0} for {len(test)} files")

    print(confusion_matrix(test[eval_args.label_col], test["pred"]))
    print(classification_report(test[eval_args.label_col], test["pred"]))
    acc = accuracy_score(test[eval_args.label_col], test["pred"])
    print(f"accuracy: {acc}")

    # save results
    ## if eval_args.save_dir does not have a csv ending, use the model name + csv
