"""Convert audiofiles to melspectrograms for analysis using CNNs. We convert the raw audiofile and leave augmentations to the training function"""

import librosa
from librosa.core.spectrum import amplitude_to_db
import torch
import torchaudio
from torchaudio import transforms
from torchaudio.transforms import MelSpectrogram

from src.make_windows import stack_frames


class AudioProcessor:
    """Processess wav files to melspectrograms"""

    def __init__(
        self,
        target_sampling_rate: int = 16000,
        window_length: int = 5,
        stride_length: int = 1,
        n_mels=128,
    ) -> None:
        self.target_sampling_rate = target_sampling_rate
        self.window_length = window_length
        self.stride_length = stride_length
        self.n_mels = n_mels

        self.melspectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sampling_rate, n_mels=n_mels
        )

    def _speech_file_to_array(self, path):
        "resample audio to match what the model expects (16000 khz)"
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(
            sampling_rate, self.target_sampling_rate
        )
        speech = resampler(speech_array).squeeze()
        return speech

    def make_windowed_mel_spectrograms(self, path):
        audio = self._speech_file_to_array(path)
        windowed_audio = stack_frames(
            audio, self.target_sampling_rate, self.window_length, self.stride_length
        )
        melspectrograms = [self.melspectrogrammer(audio) for audio in windowed_audio]
        return melspectrograms

    def _audio_to_melspectrogram(self, audio):

        return torchaudio.transforms.MelSpectrogram(audio)


WINDOW_LENGTH = 5
STRIDE_LENGTH = 1

from pathlib import Path

filepath = (
    Path().cwd()
    / "data"
    / "multi_diagnosis"
    / "DEPR"
    / "depression_1ep"
    / "full_denoise"
)

files = list(filepath.glob("*.wav"))

proc = AudioProcessor()


test = proc._speech_file_to_array(files[10])
stacked_test = stack_frames(test, 16000, 5, 1)

melspectrogram = torch.nn.Sequential(
    transforms.MelSpectrogram(), transforms.AmplitudeToDB()
)

transform = MelSpectrogram(16000)

amplitude_to_db

import librosa
import librosa.display
import matplotlib.pyplot as plt

# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

librosa.display.specshow(
    melspectrogram(test).numpy(), x_axis="time", y_axis="mel", sr=16000, fmax=6000
)


x = amplitude_to_db(transform(test).numpy())

transform(test)

test = proc.make_windowed_mel_spectrograms(files[0])
