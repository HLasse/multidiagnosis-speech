"""evaluate model performance
TODO
- Figure out why the pandas version and HF version do not give the same results.
- Pandas one seems to be the most trustworthy (according to metrics on wandb + general knowledge)
- Try to predict something in the preprocess function (so one map is enough)
- Turn this into a class instead to clean it up
"""

import dataclasses
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoConfig, HfArgumentParser, Wav2Vec2FeatureExtractor

from src.baseline_utils.baseline_pl_model import BaselineClassifier
from src.processor import CustomWav2Vec2Processor
from src.wav2vec_model import Wav2Vec2ForSequenceClassification
from src.make_windows import stack_frames
from src.wav2vec_model import Wav2Vec2ForSequenceClassification
from src.processor import CustomWav2Vec2Processor


@dataclass
class EvalArguments:
    model_type: str = field(
        metadata={
            "help": "Model type: either 'wav2vec', 'embedding_baseline' or 'cnn_baseline'"
        }
    )
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
        default="results",
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
    performance_by_id: bool = field(
        default=True, metadata={"help": "evaluate performance by participant"}
    )
    metadata_path: Optional[str] = field(
        default=None, metadata={"help": "path to metadata if merging is desired"}
    )
    num_classes: int = field(default=4, metadata={"help": "number of classes"})
    feature_set: str = field(
        default=None, metadata={"Which feature set is used (baseline models only)"}
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
        sampling_rate=target_sampling_rate,
        frame_length=eval_args.window_length,
        frame_stride=eval_args.stride_length,
    )

    # if windowed_arrays.shape[1] != 64000:
    #     print("debug")
    return windowed_arrays


def preprocess_stacked_speech_files(batch):
    """Process batch of audio files into windows of io_args.window_length with io_args.stride_length
    and return input values as well as metadata for the batch"""
    speech_list = [
        stack_speech_file_to_array(path) for path in batch[eval_args.input_col]
    ]
    labels = [config.label2id[label] for label in batch[eval_args.label_col]]
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


def preprocess_single_file(batch):
    "preprocess hf dataset/load data"
    speech_list = [speech_file_to_array_fn(path) for path in batch[eval_args.input_col]]
    labels = [config.label2id[label] for label in batch[eval_args.label_col]]

    out = processor(speech_list, sampling_rate=target_sampling_rate)
    out["labels"] = list(labels)
    return out


def predict(batch):
    input_values = torch.FloatTensor(batch["input_values"]).to(device)
    attention_mask = torch.FloatTensor(batch["attention_mask"]).to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    scores = F.softmax(logits).detach().cpu().numpy()
    pred = [config.id2label[np.argmax(score)] for score in scores]
    confidence = scores[np.arange(scores.shape[0]), np.argmax(scores, axis=1)]
    batch["prediction"] = pred
    batch["confidence"] = confidence
    batch["scores"] = scores
    return batch


def wav2vec_predict_file(speech, sampling_rate):
    features = processor(
        speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    return logits


def predict_file(path, sampling_rate):
    """Predict single file"""
    speech = speech_file_to_array_fn(path, sampling_rate)
    if eval_args.model_type == "wav2vec":
        logits = wav2vec_predict_file(speech, sampling_rate)
    else:
        with torch.no_grad():
            logits = model(speech)

    scores = F.softmax(logits, dim=0).detach().cpu().numpy()[0]
    pred = config.id2label[np.argmax(scores)]
    confidence = scores[np.argmax(scores)]
    return pred, confidence, scores


def wav2vec_predict_windows(speech_windows: np.array, sampling_rate):
    features = processor(
        speech_windows, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    # needs to remove first dimension if more than one window
    # squeeze doesn't work if only 1 window (removes two dimensions)
    input_values = features.input_values.to(device).flatten(0, 1)
    attention_mask = features.attention_mask.to(device).flatten(0, 1)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    return logits


def predict_windows(path, sampling_rate, aggregation_fn=lambda x: np.mean(x, axis=0)):
    """Create windows from an input file and output aggregated predictions"""
    speech_windows = stack_speech_file_to_array(path)
    if eval_args.model_type == "wav2vec":
        logits = wav2vec_predict_windows(speech_windows, sampling_rate)
    else:
        with torch.no_grad():
            logits = model(speech_windows)

    # check if this is correct (might need to remove dim=0)
    scores = [F.softmax(logit, dim=0).detach().cpu().numpy() for logit in logits]
    pooled_pred = aggregation_fn(scores)
    pred = config.id2label[np.argmax(pooled_pred)]
    confidence = pooled_pred[np.argmax(pooled_pred)]
    return pred, confidence, pooled_pred


def add_predicted_and_confidence(df):
    if eval_args.use_windowing:
        pred, confidence, scores = predict_windows(
            df[eval_args.input_col], target_sampling_rate
        )
    else:
        pred, confidence, scores = predict_file(
            df[eval_args.input_col], target_sampling_rate
        )
    df["prediction_audio"] = pred
    df["confidence"] = confidence
    df["scores"] = scores
    return df


def print_performance(df, label_col, prediction_col):
    """Print model performance"""
    print(confusion_matrix(df[label_col], df[prediction_col]))
    print(classification_report(df[label_col], df[prediction_col]))
    acc = accuracy_score(df[label_col], df[prediction_col])
    print(f"Accuracy: {acc}")


def calculate_grouped_performance(df):
    """Calculates performance on a grouped variable, assuming 'scores' contains
    softmaxed model output"""
    scores = np.array(df["scores"].tolist())
    # calculating average score per group (return this too?)
    mean_scores = scores.mean(axis=0)
    prediction = np.argmax(mean_scores)

    return {
        "prediction_grouped": config.id2label[prediction],
        eval_args.label_col: df[eval_args.label_col].unique()[0],
    }


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

    if eval_args.model_type == "wav2vec":
        config = AutoConfig.from_pretrained(eval_args.model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            eval_args.model_path
        )
        processor = CustomWav2Vec2Processor(feature_extractor=feature_extractor)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            eval_args.model_path
        ).to(device)
    elif eval_args.model_type in ["embedding_baseline", "cnn_baseline"]:
        model = BaselineClassifier.load_from_checkpoint(
            eval_args.model_path,
            num_classes=eval_args.num_classes,
            feature_set=eval_args.feature_set,
            learning_rate=0,
            train_loader=None,
            val_loader=None,
        )
    else:
        raise SyntaxError(
            f"{eval_args.model_type} not a valid model type. Use either 'wav2vec', 'embedding_baseline', or 'cnn_baseline'"
        )

    target_sampling_rate = 16000

    model.eval()
    # print("[INFO] Loading dataset...")
    # dataset = load_dataset(
    #     "csv", data_files={"test": eval_args.data_path}, delimiter=","
    # )
    # test = dataset["test"]

    # print("[INFO] Preprocessing dataset...")
    # if eval_args.use_windowing:
    #     print(
    #         f"[INFO] Using windows of size {eval_args.window_length} and stride {eval_args.stride_length}"
    #     )
    #     test = test.map(
    #         preprocess_stacked_speech_files,
    #         batched=True,
    #         remove_columns=dataset["test"].column_names,
    #     )
    # else:
    #     print("[INFO] Not applying windowing")
    #     test = test.map(preprocess_single_file, batched=True)

    # print("[INFO] Applying model...")
    # test = test.map(predict, batched=True, batch_size=300)

    # test = test.to_pandas()

    # # Merge with metadata (if specified)
    # if eval_args.metadata_path is not None:
    #     metadata = pd.read_csv(eval_args.metadata_path)
    #     metadata = metadata.rename(columns={"ID": "id"})
    #     test = pd.merge(test, metadata, validate="many_to_one", on="id")

    # # Save results to csv
    # save_dir = Path(eval_args.save_dir)
    # if not save_dir.exists():
    #     save_dir.mkdir()
    # filename = Path(eval_args.model_path).name + "eval.csv"
    # save_path = save_dir / filename
    # print(f"[INFO] Saving results to {save_path}")
    # test.to_csv(save_dir / filename, index=False)

    # # Evaluate performance by window
    # if eval_args.performance_by_window:
    #     print(f"[INFO] Performance by window:")
    #     print_performance(test, eval_args.label_col, "prediction")

    # if eval_args.performance_by_file:
    #     print(f"[INFO] Performance by file:")
    #     test_by_file = (
    #         test.groupby("file").apply(calculate_grouped_performance).apply(pd.Series)
    #     )
    #     print_performance(test_by_file, eval_args.label_col, "prediction_grouped")

    # if eval_args.performance_by_id:
    #     print(f"[INFO] Performance by id:")
    #     test_by_id = (
    #         test.groupby("id").apply(calculate_grouped_performance).apply(pd.Series)
    #     )
    #     print_performance(test_by_id, eval_args.label_col, "prediction_grouped")

    ### all pandas
    test = pd.read_csv(eval_args.data_path)
    print(f"Evaluating on {eval_args.data_path} containing {len(test)} files")
    # apply predictions
    t0 = time.time()

    # register progress apply with tqdm
    tqdm.pandas()
    # run predictions
    test = test.progress_apply(add_predicted_and_confidence, axis=1)
    # test = test.apply(add_predicted_and_confidence, axis=1)
    print(f"Time taken: {time.time() - t0} for {len(test)} files")

    # Save to csv
    save_dir = Path(eval_args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    filename = Path(eval_args.model_path).name + "eval.csv"
    save_path = save_dir / filename
    print(f"[INFO] Saving results to {save_path}")
    test.to_csv(save_dir / filename, index=False)

    print("[INFO] Performance by file:")
    print_performance(test, eval_args.label_col, "prediction_audio")

    if eval_args.performance_by_id:
        print(f"[INFO] Performance by id:")
        test_by_id = (
            test.groupby("id").apply(calculate_grouped_performance).apply(pd.Series)
        )
        print_performance(test_by_id, eval_args.label_col, "prediction_grouped")
