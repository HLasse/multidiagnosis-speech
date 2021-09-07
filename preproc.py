import librosa
import numpy as np
import torchaudio
import os
import datasets

import json

data = os.path.join(os.getcwd(), "data", "Neutre")

filepaths = os.listdir(data)
file_dict = [{"file": os.path.join(data, path)} for path in filepaths]

# save to json file to we can use hugginface datasets loader
# seems pretty stupid
for i, file in enumerate(file_dict):
     with open(f"json_files/sample_{i}.json", "w") as outfile:
         json.dump(file, outfile)


def get_gender(filepath):
    """files are coded so first two characters decide gender.
    odd = man, even = woman
    code woman as 1, men as 0"""
    if int(filepath.split("/")[-1][0:2]) % 2 == 0:
        return 1
    return 0


# Loader for datafiles
def speech_file_to_array(batch):
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["label"] = get_gender(batch["file"])
    return batch

# resample audio to 16 khz
def resample(batch, target_sr = 16_000):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), batch["sampling_rate"], target_sr)
    batch["sampling_rate"] = target_sr
    return batch

if __name__ == "__main__":

    dataset = datasets.load_dataset("json", data_files=[os.path.join("json_files", file) for file in os.listdir("json_files")], split="train")

    # process dataset
    dataset = dataset.map(speech_file_to_array)
    dataset = dataset.map(resample)

    dataset.save_to_disk("preproc_data")