
import torch

from torchaudio.transforms import MelSpectrogram
from speechbrain.pretrained import EncoderClassifier
from speechbrain.lobes.features import MFCC

import numpy as np

import opensmile

def get_embedding_fns():
    """Helper to return a dict of embedding names and corresponding functions

    Arguments:
        embedding_type {str} -- [description]

    Returns:
        [type] -- [description]
    """
    

    xvector_embedding = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )

    def xvector_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 512)
        if isinstance(audio, np.ndarray):
            audio=torch.tensor(audio)
        return xvector_embedding.encode_batch(audio).squeeze()

    egemapsv2 = opensmile.Smile(
             feature_set=opensmile.FeatureSet.eGeMAPSv02,
             feature_level=opensmile.FeatureLevel.Functionals,
         )

    def egemaps_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 88)
        embeddings = [egemapsv2.process_signal(a, sampling_rate=16000).to_numpy().squeeze() for a in audio]
        return torch.tensor(embeddings)


    compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=10,
    )

    def compare_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 6373)
        embeddings = [compare.process_signal(a, sampling_rate=16000).to_numpy().squeeze() for a in audio]
        return torch.tensor(embeddings)


    mel_extractor = MelSpectrogram(sample_rate=16000, n_mels=128)

    def aggregated_mfccs_fn(audio) -> torch.tensor:
        # shape = (batch, 128)
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).type(torch.float)
        mfccs = mel_extractor(audio)
        return torch.mean(mfccs, 2)


    def windowed_mfccs_fn(audio):
        # shape = (batch, n_mels, samples (401)))
        return mel_extractor(audio)



    return {
        "xvector" : xvector_embedding_fn,
        "egemaps" : egemaps_embedding_fn,
        "compare" : compare_embedding_fn,
        "aggregated_mfccs" : aggregated_mfccs_fn,
        "windowed_mfccs" : windowed_mfccs_fn,
        }




