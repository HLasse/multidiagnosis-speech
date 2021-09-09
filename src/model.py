"""XLSR/Wav2Vec2.0 with classification head on top"""

import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel, 
    Wav2Vec2Model
)


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for classification tasks
        Layers:
        - dropout
        - dense layer (default xlsr hidden size = 1024)
        - relu
        - dropout
        - classificiation layer of size num_labels
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.hidden_dropout(features)
        x = F.relu(self.dense(features))
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 Model for speech classification, similar to BertForSequenceClassification"""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.wav2vec = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        """The feature extractor (the CNN layers) are sufficiently trained during pre-training.
        As recommended by the paper, we do not finetune them further.
        the wav2vec.freeze_feature_extractor method essentially just sets `requires_grad` to `False`
        for all trainable parameters in the feature extractor"""
        self.wav2vec.freeze_feature_extractor()

    def merge_hidden_states(self, hidden_states):
        return torch.mean(hidden_states)

    def forward(self,
                input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, # whether to return a SequenceClassifierOutput or just a dict
                labels=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec(input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            )

        print("test")
        
