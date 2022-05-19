"""Read all audio files and create windowed data with stride. Saves as arrow format to save memory"""
import datasets
from datasets import load_dataset
import sys
sys.path.append("/work/wav2vec_finetune")
from src.make_windows import stack_frames

import torchaudio

from pathlib import Path

def stack_speech_file_to_array(path):
    """Loads and resamples audio to target sampling rate and converts the
    audio into windows of specified length and stride"""
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, TARGET_SAMPLING_RATE)
    speech_array = resampler(speech_array)

    windowed_arrays = stack_frames(
        speech_array.squeeze(),
        sampling_rate=TARGET_SAMPLING_RATE,
        frame_length=WINDOW_LENGTH,
        frame_stride=STRIDE_LENGTH,
    )
    return windowed_arrays


def preprocess_stacked_speech_files(batch):
    """Process batch of audio files into windows of io_args.window_length with io_args.stride_length
    and return input values as well as metadata for the batch"""
    speech_list = [
        stack_speech_file_to_array(path) for path in batch[INPUT_COL]
    ]
    n_windows = [len(window) for window in speech_list]

    # make new larger dictionary that contains the flattened values
    # labels = label as idx
    out = {"audio": []}
    # save metadata from other columns
    for meta_key in batch.keys():
        out[meta_key] = []
    # looping through list of processed stacked speech arrays
    for i, speech_window in enumerate(speech_list):
        # un-nesting the stacked time windows
        out["audio"].extend(speech_window.astype("float64"))
        # adding metadata to be able to reidentify files
        for meta_key, meta_value in batch.items():
            out[meta_key] += [meta_value[i]] * n_windows[i]

    return out


if __name__ == "__main__":

    TARGET_SAMPLING_RATE = 16000
    WINDOW_LENGTH = 5
    STRIDE_LENGTH = 1
    INPUT_COL = "file"

    DATA_DIR = Path("data") / "audio_file_splits" 
    BASE_SAVE_DIR = DATA_DIR / "windowed_splits"
    if not BASE_SAVE_DIR.exists():
        BASE_SAVE_DIR.mkdir()

    data_files = {
            "train": str(DATA_DIR / "audio_train_split.csv"),
            "validation": str(DATA_DIR / "audio_val_split.csv"),
            "test" : str(DATA_DIR / "audio_test_split.csv")
        }

    print("[INFO] Loading dataset...")
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")

    print(
        f"[INFO] Using windows of size {WINDOW_LENGTH} and stride {STRIDE_LENGTH}"
    )

    dataset = dataset.map(
        preprocess_stacked_speech_files,
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=200
    )

    dataset["train"].to_parquet(BASE_SAVE_DIR / "windowed_train_split.parquet")
    dataset["validation"].to_parquet(BASE_SAVE_DIR / "windowed_val_split.parquet")
    dataset["test"].to_parquet(BASE_SAVE_DIR / "windowed_test_split.parquet")
