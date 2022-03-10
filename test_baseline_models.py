import pandas as pd

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.baseline_utils.dataloader import MultiDiagnosisDataset

import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from speechbrain.pretrained import EncoderClassifier
from speechbrain.lobes.features import MFCC

import opensmile

import torchmetrics

files = "data/audio_file_splits/dummy_train_set.csv"

train = pd.read_csv(files)
train = train[train["duration"] >= 5]

mapping = {"ASD": 0, "DEPR": 1, "SCHZ": 2, "TD": 3}


def label2idx(label):
    mapping = {"ASD": 0, "DEPR": 1, "SCHZ": 2, "TD": 3}
    return mapping[label]


train["label_id"] = train.label.replace(mapping)


FEATURE_SET = "aggregated_mfccs"

if FEATURE_SET == "xvector":
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )

    def embedding_fn(audio):
        # shape = (batch, 512)
        return classifier.encode_batch(audio).squeeze()


if FEATURE_SET == "opensmile":
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    def embedding_fn(audio):
        # shape = (batch, 88)
        return smile.process_signal(audio, sampling_rate=16000).to_numpy().squeeze()


if FEATURE_SET == "aggregated_mfccs":
    mel_extractor = MelSpectrogram(sample_rate=16000, n_mels=128)

    def embedding_fn(audio):
        # shape = (batch, 128)
        mfccs = mel_extractor(audio)
        return torch.mean(mfccs, 1)


if FEATURE_SET == "windowed_mfccs":
    mel_extractor = MelSpectrogram(sample_rate=16000, n_mels=128)

    def embedding_fn(audio):
        # shape = (batch, n_mels, samples (401)))
        return mel_extractor(audio)


class BaselineClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, feature_set: str):
        super(BaselineClassifier, self).__init__()

        input_dims = {"xvector": 512, "opensmile": 88, "aggregated_mfccs": 128}

        self.linear = nn.Linear(input_dims[feature_set], 1028)
        self.classifier = nn.Linear(1028, num_classes)
        self.stat_scores = torchmetrics.classification.StatScores()

    def forward(self, x):
        x = F.leaky_relu(self.linear(x))
        return self.classifier(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.stat_scores(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, val_epoch):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_epoch_end(self, outs) -> None:
        self.log("train_stat_scores", self.stat_scores)


# model with aggregated mels and a CNN on the samples
training_data = MultiDiagnosisDataset(
    train["file"], train["label_id"], embedding_fn=embedding_fn
)
val_data = MultiDiagnosisDataset(
    train["file"], train["label_id"], embedding_fn=embedding_fn
)

train_loader = DataLoader(training_data, batch_size=2)
val_loader = DataLoader(val_data, batch_size=2)

model = BaselineClassifier(num_classes=3, feature_set=FEATURE_SET)
trainer = pl.Trainer(max_epochs=200)

trainer.fit(model, train_loader, val_loader)

# MultiDiagnosisDataset()

