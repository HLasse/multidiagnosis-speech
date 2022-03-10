"""Script to train baseline models:
1. Linear layer on pretrained x-vector
2. X-vector trained from scratch on data
3. Classifier trained on MFCCs (speechbrain MFCC/Fbank)
4. Classifier trained on opensmile features


Use Pyannote training procedure:
    - TODO Define a custom Task for speech classification 
      (one for multi-class and one for binary classification)
    - TODO Create pytorch models (as subclass of pyannote Model class) 
      for each model type above
"""

from typing import Union, Optional

import torch
import torchaudio

import torch_audiomentations

import wandb

import dataclasses

