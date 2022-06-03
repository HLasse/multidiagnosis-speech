"""Read all audio files and create windowed data with stride. Saves as arrow format to save memory"""
import datasets
from datasets import load_dataset
import sys
import torch

sys.path.append("/work/wav2vec_finetune")
from src.make_windows import stack_frames
from constants import WINDOW_SIZE, WINDOW_STRIDE

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
        frame_length=WINDOW_SIZE,
        frame_stride=WINDOW_STRIDE,
    )
    return windowed_arrays


def preprocess_stacked_speech_files(batch):
    """Process batch of audio files into windows of io_args.window_length with io_args.stride_length
    and return input values as well as metadata for the batch"""
    speech_list = [stack_speech_file_to_array(path) for path in batch[INPUT_COL]]
    n_windows = [len(window) for window in speech_list]

    # make new larger dictionary that contains the flattened values
    # labels = label as idx
    out = {"audio": []}
    # save metadata from other columns
    for meta_key in batch.keys():
        out[meta_key] = []
        out["file_segment_n"] = []
    # looping through list of processed stacked speech arrays
    for i, speech_window in enumerate(speech_list):
        # un-nesting the stacked time windows and converting to 32 bit float instead of 64
        out["audio"].extend(speech_window.astype("float32"))
        # adding metadata to be able to reidentify files
        for meta_key, meta_value in batch.items():
            out[meta_key] += [meta_value[i]] * n_windows[i]
        # Adding segment identifier
        for n in range(n_windows[i]):
            out["file_segment_n"] += [n]

    return out


def audio_to_file(example):
    f = Path(example["file"])
    origin = f.parent.parent.parent.name
    label = f.parent.parent.name
    filename = f.stem
    filename = BASE_SAVE_DIR / f"{origin}_{label}_{filename}_{example['file_segment_n']}.wav"
    # Saving file (adding an empty 'channel' dimension)
    torchaudio.save(str(filename), example["audio"][None,:], sample_rate=TARGET_SAMPLING_RATE)
    example["filename"] = str(filename)
    return example


if __name__ == "__main__":

    TARGET_SAMPLING_RATE = 16000
    INPUT_COL = "file"

    DATA_DIR = Path("data") / "audio_file_splits"
    BASE_SAVE_DIR = Path("data") / "windowed_data"
    if not BASE_SAVE_DIR.exists():
        BASE_SAVE_DIR.mkdir()
    SPLIT_SAVE_DIR = Path("data") / "audio_file_splits" / "windowed_splits"
    if not SPLIT_SAVE_DIR.exists():
        SPLIT_SAVE_DIR.mkdir()

    data_files = {
        "train": str(DATA_DIR / "audio_train_split.csv"),
        "validation": str(DATA_DIR / "audio_val_split.csv"),
        "test": str(DATA_DIR / "audio_test_split.csv"),
    }

    print("[INFO] Loading dataset...")
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")

    print(f"[INFO] Using windows of size {WINDOW_LENGTH} and stride {STRIDE_LENGTH}")

    dataset = dataset.map(
        preprocess_stacked_speech_files,
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=200,
    )
    audio_format = {
    "type": "torch",
    "format_kwargs": {"dtype": torch.float32},
    "columns": ["audio"],
    }
    dataset.set_format(**audio_format, output_all_columns=True)


    dataset = dataset.map(
        audio_to_file,
        batch_size=200
    )

    dataset = dataset.remove_columns("audio")
    dataset["train"].to_csv(SPLIT_SAVE_DIR / "windowed_train_split.csv")
    dataset["validation"].to_csv(SPLIT_SAVE_DIR / "windowed_validation_split.csv")
    dataset["test"].to_csv(SPLIT_SAVE_DIR / "windowed_test_split.csv")
    