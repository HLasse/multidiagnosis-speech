from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import Dataset
import torchaudio

from datasets import Dataset
from typing import Callable, Union, List
import pandas as pd

import numpy as np


class MultiDiagnosisDataset(Dataset):
    def __init__(
        self,
        data: Dataset,  # should contain an "audio" and "label_id" column
        augment_fn: Callable = None,
        embedding_fn: Callable = None,
        sample_rate: int = 16000,
        window_size: int = 5,
    ):
        self.data = data
        self.augment_fn = augment_fn
        self.embedding_fn = embedding_fn
        self.sample_rate = sample_rate
        self.window_size = window_size

        weights = compute_class_weight(class_weight="balanced")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = self.data["audio"][idx]
        if self.augment_fn:
            audio = self.augment_fn(audio)
        if self.embedding_fn:
            audio = self.embedding_fn(audio)
        label = self.data["label_idx"][idx]
        return audio, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_files = "data/audio_file_splits/audio_train_split.csv"
    train = pd.read_csv(train_files)
    train = train[train["duration"] >= 5]
    mapping = {"ASD": 0, "DEPR": 1, "SCHZ": 2, "TD": 3}
    train["label_id"] = train.label.replace(mapping)

    diagnosis = "ASD"
    train_set = train[train["origin"] == diagnosis]

    dataset = MultiDiagnosisDataset(
        train_set["file"].tolist(), train_set["label_id"].tolist()
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    dat_load = iter(dataloader)
    for i in range(100):
        x, labs = next(dat_load)
