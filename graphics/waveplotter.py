import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from torch_audiomentations import Compose, Gain, AddColoredNoise

long_file = "Study1S101T4.wav"
short_file = "ASD_ASD_Study1S101T4_0.wav"
short_file1 = "ASD_ASD_Study1S101T4_1.wav"


def plot_and_save(file, color):
    y, sr = librosa.load(file)
    librosa.display.waveshow(y, sr=sr, color=color)
    plt.ylim((-0.8, 0.8))
    plt.savefig(f"{file}.png")
    plt.close()


for f, c in zip(
    [long_file, short_file, short_file1], [None, "firebrick", "forestgreen"]
):
    plot_and_save(f, c)


sf1, sr = librosa.load(short_file)

aug = Compose(
    transforms=[
        Gain(
            min_gain_in_db=2, max_gain_in_db=5, p=1  # prob set to -3  # prob set to 4
        ),
        AddColoredNoise(
            p=1,
            min_snr_in_db=12,  # probably use 7-30 in script
            max_snr_in_db=13,
            max_f_decay=-1,
            min_f_decay=-1.1,
        ),
    ]
)


sf1_tensor = torch.tensor(sf1)[None, None, :]
sf1_aug = aug(sf1_tensor, sample_rate=sr)

sf1_aug = sf1_aug.numpy().squeeze()

librosa.display.waveshow(sf1_aug, sr=sr, color="firebrick")
plt.ylim((-0.8, 0.8))
plt.savefig("augmented.png")
plt.close()
