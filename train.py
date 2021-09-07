from platform import processor
import librosa
import numpy as np
import torchaudio
import os
import datasets

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification

dataset = datasets.load_from_disk("preproc_data")

# Load feature extractor:
## params:
# feature_size: always one as it's the raw signal (so 1 feature == the signal)
# sapling_rate: obvious
# padding_value: which value to pad short inputs with
# do_normalize: whether to normalize (scale) the input
# return_attention_mask: whether to use attention mask. (yes)
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# load model (without any specific head on top)
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# load model (with a simple linear linear (input 1024 -> 256 units) and a binary classification on top)
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# https://github.com/pytorch/fairseq/issues/3006
## Might be better to only use the encoding (so cut before the transformer) and own layers (e.g. lstm) for training the classifier
## wav1vec 1.0 is also worth a try. 