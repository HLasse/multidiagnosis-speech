"""Class to evaluate models.
TODO
    -  Turn into a superclass with an embedding and wav2vec subclass"""

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

from .baseline_utils.baseline_pl_model import BaselineClassifier
from .processor import CustomWav2Vec2Processor
from .wav2vec_model import Wav2Vec2ForSequenceClassification
from .make_windows import stack_frames
from .wav2vec_model import Wav2Vec2ForSequenceClassification
from .processor import CustomWav2Vec2Processor
from .baseline_utils.embedding_fns import get_embedding_fns

from .util import create_argparser


class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        save_dir: str = "results",
        data_path: str = "data/audio_file_splits/audio_val_split.csv",
        input_col: str = "file",
        label_col: str = "label",
        sampling_rate: int = 16_000,
        use_windowing: bool = True,
        window_length: int = 5,
        stride_length: float = 1,
        num_classes: int = 4,
        id2label: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.input_col = input_col
        self.label_col = label_col
        self.save_dir = save_dir
        self.sampling_rate = sampling_rate
        self.use_windowing = use_windowing
        self.window_length = window_length
        self.stride_length = stride_length
        self.num_classes = num_classes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # if self.model_type == "wav2vec":
        #     self._setup_wav2vec_model()
        # if self.model_type == "embedding_baseline":
        #     self._setup_embedding_baseline()
        # self.model.eval()

        self.df = pd.read_csv(data_path)

        ### check if this matches the config from wav2vec!!!!!
        if not id2label:
            if num_classes != 4:
                raise ValueError("id2label must be set if less than 4 classes")
            self.id2label = {0: "ASD", 1: "DEPR", 2: "SCHZ", 3: "TD"}
        else:
            self.id2label = id2label

        # if feature_set:
        #     embedding_fns = get_embedding_fns()
        #     self.embedding_fn = embedding_fns[feature_set]

        tqdm.pandas()

    def evaluate_model(self):
        t0 = time.time()
        print(f"Evaluting on {self.data_path} containing {len(self.df)} files.")
        self.df = self.df.progress_apply(self.add_predicted_and_confidence, axis=1)
        print(f"Time taken: {time.time() - t0}")


    def setup_model(self):
        pass

    # def _setup_wav2vec_model(self):
    #     """Loads the necesarry components for evaluating wav2vec models"""
    #     self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    #         "facebook/wav2vec2-xls-r-300m"
    #     )
    #     self.processor = CustomWav2Vec2Processor(
    #         feature_extractor=self.feature_extractor
    #     )
    #     self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
    #         self.model_path
    #     ).to(self.device)

    # def _setup_embedding_baseline(self):
    #     self.model = BaselineClassifier.load_from_checkpoint(
    #         self.model_path,
    #         num_classes=self.num_classes,
    #         feature_set=self.feature_set,
    #         learning_rate=0,
    #         train_loader=None,
    #         val_loader=None,
    #     )

    def add_predicted_and_confidence(self, df):
        if self.use_windowing:
            pred, confidence, scores = self._predict_windows(df[self.input_col])
        else:
            pred, confidence, scores = self._predict(df[self.input_col])

        df["prediction_audio"] = pred
        df["confidence_audio"] = confidence
        df["scores_audio"] = scores
        return df

    def _predict_windows(self, path: str):
        """Use model on a windowed audio file"""
        speech_windows = self.stack_speech_file_to_array(path)
        logits = 
        if self.model_type == "wav2vec":
            logits = self._wav2vec_predict_windows(speech_windows)
        if self.model_type == "embedding_baseline":
            logits = self._embedding_predict_windows(speech_windows)

        scores = [F.softmax(logit, dim=0).detach().cpu().numpy() for logit in logits]
        pooled_pred = np.mean(scores, axis=0)
        pred = self.id2label[np.argmax(pooled_pred)]
        confidence = pooled_pred[np.argmax(pooled_pred)]
        return pred, confidence, pooled_pred

    def _wav2vec_predict_windows(self, speech_windows: np.array):
        features = self.processor(
            speech_windows,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        # needs to remove first dimension if more than one window
        # squeeze doesn't work if only 1 window (removes two dimensions)
        input_values = features.input_values.to(device).flatten(0, 1)
        attention_mask = features.attention_mask.to(device).flatten(0, 1)

        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        return logits

    def _embedding_predict_windows(self, speech_windows):
        features = self.embedding_fn(speech_windows)
        # If only a single file in the batch we need to add the batch dimension
        if len(features.shape) == 1:
            features = features.unsqueeze(dim=0)
        with torch.no_grad():
            logits = self.model(features)
        # baseline models return log softmax - removing the log
        logits = torch.exp(logits)
        return logits

    def predict_file(self):
        pass

    def stack_speech_file_to_array(self, path: str) -> np.array:
        """Load and resample and audio file. Returns the file as a windowed array

        Arguments:
            path {str} -- Path to file

        Returns:
            np.array -- windowed audio
        """
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
        speech_array = resampler(speech_array)

        windowed_arrays = stack_frames(
            speech_array.squeeze(),
            sampling_rate=self.sampling_rate,
            frame_length=self.window_length,
            frame_stride=self.stride_length,
        )
        return windowed_arrays

    def save_to_csv(self, filename):
        save_dir = Path(self.save_dir)
        if not save_dir.exists():
            save_dir.mkdir()
        save_path = save_dir / filename
        print(f"[INFO] Saving results to {save_path}")
        self.df.to_csv(save_dir / filename, index=False)

    def _print_performance(self, df, prediction_col):
        """Print model performance"""
        print("[INFO] Performance by file:")
        print(confusion_matrix(df[self.label_col], df[prediction_col]))
        print(classification_report(df[self.label_col], df[prediction_col]))
        acc = accuracy_score(df[self.label_col], df[prediction_col])
        print(f"Accuracy: {acc}")

    def print_performance(self):
        self._print_performance(self.df, "prediction_audio")

    def print_performance_by_id(self):
        print(f"[INFO] Performance by id:")
        self.df_by_id = (
            self.df.groupby("id")
            .apply(self._calculate_grouped_performance)
            .apply(pd.Series)
        )
        self._print_performance(self.df_by_id, "prediction_grouped")

    def _calculate_grouped_performance(self, df):
        """Calculates performance on a grouped variable, assuming 'scores' contains
        softmaxed model output"""
        scores = np.array(df["scores_audio"].tolist())
        # calculating average score per group (return this too?)
        mean_scores = scores.mean(axis=0)
        prediction = np.argmax(mean_scores)

        return {
            "prediction_grouped": self.id2label[prediction],
            self.label_col: df[self.label_col].unique()[0],
        }


if __name__ == "__main__":

    import fire

    fire.Fire(ModelEvaluator)

    # yml_path = os.path.join(
    #     os.getcwd(),
    #     "configs",
    #     "eval_configs",
    #     "baseline_models",
    #     "xvector_test.yaml",
    # )
    # parser = create_argparser(yml_path)
    # arguments = parser.parse_args()

    # m_eval = ModelEvaluator(**vars(arguments))

    # m_eval.evaluate_model()
    # m_eval.print_performance()
    # m_eval.print_performance_by_id()
    # m_eval.save_to_csv("xvector_baseline_results.csv")
