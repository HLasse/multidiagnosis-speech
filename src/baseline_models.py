"""Baseline models in Pytorch"""
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.core.utils.generators import pairwise


class ClassificationModel(Model):
    """[summary]

    Args:
        embedding_fn (Optional[Callable]): [description]
        num_layers (int): [description]
        num_units (int): [description]
        num_classes (int): [description]
        task (Optional[Task], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable],
        num_layers: int,
        num_units: int,
        input_dim: int,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.num_units

        self.linear = nn.ModuleList()
        if num_layers > 1:
            self.linear = [
                nn.Linear(num_units, num_units) for i in range(num_layers - 1)
            ]
            self.linear.insert(0, nn.Linear(input_dim, num_units))
        else:
            self.linear.append(nn.Linear(input_dim, num_units))

    def build(self):
        self.classifier = nn.Linear(self.num_units, len(self.specification.classes))
        self.activation = self.default_activation()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            embedding (torch.Tensor): (batch, channel, sample)

        Returns:
            torch.Tensor: (batch, frame, classes)
        """
        x = embedding
        for linear in self.linears:
            x = F.leaky_relu(linear(x))
        return self.activation(self.classifier(x))


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_fn: Callable,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__()
        self.embedding_fn = embedding_fn

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Applies

        Args:
            waveform (torch.Tensor): (batch, channel, sample)

        Returns:
            torch.Tensor: (batch, channel, sample)
        """
        return self.embedding_fn(waveform)
