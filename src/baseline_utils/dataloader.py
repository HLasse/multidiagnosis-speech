import torch
from torch.utils.data import Dataset
import torchaudio

from typing import Callable, Union, List
import pandas as pd

import numpy as np


class MultiDiagnosisDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[List, pd.Series],
        labels: Union[List, pd.Series],
        augment_fn: Callable = None,
        embedding_fn: Callable = None,
        sample_rate: int = 16000,
        window_size: int = 5,
    ):
        self.filepaths = filepaths
        self.labels = labels
        self.augment_fn = augment_fn
        self.embedding_fn = embedding_fn
        self.sample_rate = sample_rate
        self.window_size = window_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.filepaths[idx])
        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio = resampler(audio).squeeze()

        audio_dims = audio.shape
        window_length_samples = self.window_size * self.sample_rate
        if audio_dims[0] < window_length_samples:
            raise ValueError("File duration is shorter than window size")

        # Pick a random point in the file of length window size
        maximum_offset = audio.shape[0] - window_length_samples
        if maximum_offset != 0:
            # If duration is exactly window_length seconds we get an error
            start_idx = np.random.randint(0, maximum_offset)
            audio = audio[start_idx : start_idx + window_length_samples]

        if self.augment_fn:
            audio = self.augment_fn(audio)
        if self.embedding_fn:
            audio = self.embedding_fn(audio)
        label = self.labels[idx]
        return audio, label
