import torch

from torchaudio.transforms import MelSpectrogram, MFCC
from speechbrain.pretrained import EncoderClassifier

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
            audio = torch.tensor(audio)
        return xvector_embedding.encode_batch(audio).squeeze()

    egemapsv2 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    def egemaps_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 88)
        # if only 1 file (e.g. in dataloader) just embed the 1 file
        if len(audio.shape) == 1:
            embeddings = (
                egemapsv2.process_signal(audio, sampling_rate=16000)
                .to_numpy()
                .squeeze()
            )
        else:
            embeddings = [
                egemapsv2.process_signal(a, sampling_rate=16000).to_numpy().squeeze()
                for a in audio
            ]
        return torch.tensor(np.array(embeddings))

    compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=10,
    )

    def compare_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 6373)
        if len(audio.shape) == 1:
            embeddings = (
                compare.process_signal(audio, sampling_rate=16000).to_numpy().squeeze()
            )
        else:
            embeddings = [
                compare.process_signal(a, sampling_rate=16000).to_numpy().squeeze()
                for a in audio
            ]
        return torch.tensor(embeddings)

    mfcc_extractor = MFCC(
        sample_rate=16000, n_mfcc=40, dct_type=2, norm="ortho", log_mels=False
    )

    def aggregated_mfccs_fn(audio) -> torch.tensor:
        # shape = (batch, 128)
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).type(torch.float)

        mfccs = mfcc_extractor(audio)
        if len(audio.shape) == 1:
            return torch.mean(mfccs, 1)
        else:
            return torch.mean(mfccs, 2)

    def windowed_mfccs_fn(audio):
        # shape = (batch, n_mels, samples (401)))
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).type(torch.float)
        return mfcc_extractor(audio)

    return {
        "xvector": xvector_embedding_fn,
        "egemaps": egemaps_embedding_fn,
        "compare": compare_embedding_fn,
        "aggregated_mfccs": aggregated_mfccs_fn,
        "windowed_mfccs": windowed_mfccs_fn,
    }


if __name__ == "__main__":

    sig1 = torch.rand(64000)
    sig2 = torch.rand([2, 64000])
    embs = get_embedding_fns()
    mfccs_ex = embs["aggregated_mfccs"]

    mfcc1 = mfccs_ex(sig1)
    assert mfcc1.shape[0] == 40
    mfcc2 = mfccs_ex(sig2)
    assert mfcc2.shape[1] == 40
