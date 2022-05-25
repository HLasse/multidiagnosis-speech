"""Subclassed Huggingface Trainer to allow for class weights to handle unbalanced groups"""

from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn

from transformers import (
    Trainer,
)

from torch import nn
from transformers import Trainer


class TrainerWithWeights(Trainer):
    def __init__(self, weights):
        self.super.__init__()
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=self.weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
