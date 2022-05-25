from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import Dataset
import torchaudio

from typing import Callable, Union, List
import pandas as pd

import numpy as np


class MultiDiagnosisDataset(Dataset):
    def __init__(
        self,
        paths: Union[pd.Series, List],  # path to audio files
        labels: Union[pd.Series, List],
        augment_fn: Callable = None,
        embedding_fn: Callable = None,
    ):
        self.paths = paths
        self.labels = labels
        self.augment_fn = augment_fn
        self.embedding_fn = embedding_fn

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.paths[idx])
        if self.augment_fn:
            # torch_audiomentations expects inputs of shape (batch_size, num_channels, num_samples)
            # adding a batch dimension
            if len(audio.shape) == 2:
                audio = audio[None, :, :]
            audio = self.augment_fn(audio)
            # Remove the extra dimension again
            audio = audio.squeeze()
        if self.embedding_fn:
            audio = self.embedding_fn(audio)
        label = self.labels[idx]
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
