"""evaluate model performance
TODO
- Evaluate by window and by participant (rewrite to make windows)
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchaudio

from transformers import (AutoConfig, 
    Wav2Vec2FeatureExtractor)
from src.processor import CustomWav2Vec2Processor
from src.model import Wav2Vec2ForSequenceClassification
from src.make_windows import stack_frames

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

MODEL_PATH = os.path.join("model", "xlsr_autism_stories", "checkpoint-470")
TEST = pd.read_csv(os.path.join("data", "splits", "stories_train_data_gender_False.csv"))
LABEL_COL = "Diagnosis"

USE_WINDOWING = True
WINDOW_LENGTH = 5
STRIDE_LENGTH = 1

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def stack_speech_file_to_array(path):
    speech_array, sampling_rate = torchaudio.load(path)
    windowed_arrays = stack_frames(speech_array.squeeze(), sampling_rate=sampling_rate,
                        frame_length=WINDOW_LENGTH, frame_stride=STRIDE_LENGTH)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    windowed_arrays = [resampler(window).squeeze() for window in windowed_arrays]
    return windowed_arrays

def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

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
    features = processor(speech_windows, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits    
    
    scores = [F.softmax(logit, dim=0).detach().cpu().numpy() for logit in logits]
    pooled_pred = aggregation_fn(scores)
    pred = config.id2label[np.argmax(pooled_pred)]
    confidence = pooled_pred[np.argmax(pooled_pred)]
    return pred, confidence

def add_predicted_and_confidence(df, use_windowing=USE_WINDOWING):
    if use_windowing:
        pred, confidence = predict_windows(df["file"], target_sampling_rate)
    else:
        pred, confidence = predict(df["file"], target_sampling_rate)
    df["pred"] = pred
    df["confidence"] = confidence
    return df

# setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(MODEL_PATH)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
processor = CustomWav2Vec2Processor(feature_extractor=feature_extractor)
target_sampling_rate = processor.feature_extractor.sampling_rate
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

# load test data


# apply predictions
test = TEST.apply(add_predicted_and_confidence, axis=1)

print(confusion_matrix(test[LABEL_COL], test["pred"]))
print(classification_report(test[LABEL_COL], test["pred"]))
acc = accuracy_score(test[LABEL_COL], test["pred"])
print(f"accuracy: {acc}")