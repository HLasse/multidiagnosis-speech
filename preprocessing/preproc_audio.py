"""Generate train/test split for input to HF trainer for audio files"""
import os
import pandas as pd

from pathlib import Path
import re

import librosa

SPLIT_PATH = Path("data") / "splits"
AUDIO_SPLIT_PATH = Path("data") / "audio_file_splits"


def get_metadata(filename):
    """Get id and trial from filename"""
    file = Path(filename)
    group = file.parent.parent.parent.name
    if group in ["ASD", "SCHZ"]:
        study = re.findall("Study(\d)", file.name)[0]
        id = re.findall("S(\d\d\d)", file.name)
        trial = re.findall("T(\d)", file.name)
    if group == "DEPR":
        if file.parent.parent.name == "depression_1ep":
            study = "1"
        if file.parent.parent.name == "depression_chronic":
            study = "2"
        # if control
        else:
            study = "1"
        id = re.findall("d[p]?[c]?(\d+)", file.name)
        trial = re.findall("_(\d+)", file.name)
    if not id:
        ValueError(f"No ID found in filename {file}")
    if not trial:
        ValueError(f"No trial found in filename {file}")
    return (id[0], trial[0], study)


def populate_data_dict(dirs, data_dict):
    """iterate through the files in the data directories and populate dict with meta data"""
    for diagnosis in Path(dirs).iterdir():
        if diagnosis.name == "ASD":
            files = get_files_from_dir(diagnosis / "ASD")
            data_dict = update_data_dict(data_dict, files, "ASD")
            files = get_files_from_dir(diagnosis / "TD")
            data_dict = update_data_dict(data_dict, files, "TD")
        if diagnosis.name == "DEPR":
            files = get_files_from_dir(diagnosis / "depression_1ep")
            data_dict = update_data_dict(data_dict, files, "DEPR")
            files = get_files_from_dir(diagnosis / "depression_chronic")
            data_dict = update_data_dict(data_dict, files, "DEPR")
            files = get_files_from_dir(diagnosis / "TD")
            data_dict = update_data_dict(data_dict, files, "TD")
        if diagnosis.name == "SCHZ":
            files = get_files_from_dir(diagnosis / "schizophrenic")
            data_dict = update_data_dict(data_dict, files, "SCHZ")
            files = get_files_from_dir(diagnosis / "TD")
            data_dict = update_data_dict(data_dict, files, "TD")
    # flatten
    data_dict = {k: flatten(v) for k, v in data_dict.items()}
    return data_dict


def update_data_dict(data_dict, files, label):
    """update metadata dictionary based on information in filename"""
    data_dict["file"].append(files)
    data_dict["label"].append([label] * len(files))
    data_dict["origin"].append([files[0].split("/")[-4]] * len(files))
    ids, trials, study = zip(*[get_metadata(f) for f in files])
    data_dict["id"].append(ids)
    data_dict["trial"].append(trials)
    data_dict["study"].append(study)
    return data_dict


def get_files_from_dir(path):
    """get files of interest"""
    wd = path / "full_denoise"
    files = os.listdir(wd)
    files = [os.path.join(wd, f) for f in files]
    return files


def flatten(t):
    """flatten list"""
    return [item for sublist in t for item in sublist]


def create_unique_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new IDs by adding the origin, label, and study to the id

    Args:
        df (pd.DataFrame): DataFrame with id, label, and origin columns

    Returns:
        pd.DataFrame: Original dataframe with updated id column
    """

    df["id"] = df["origin"] + "_" + df["label"] + "_" + df["study"] + "_" + df["id"]
    return df


def audiofile_duration(path):
    """Get duration of file in seconds"""
    return librosa.get_duration(filename=path)


def read_train_val_test():
    train_ids = pd.read_csv(SPLIT_PATH / "train_split.csv")
    val_ids = pd.read_csv(SPLIT_PATH / "validation_split.csv")
    test_ids = pd.read_csv(SPLIT_PATH / "test_split.csv")
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    data_dirs = Path("data") / "multi_diagnosis"
    data_dict = {
        "file": [],
        "label": [],
        "origin": [],
        "id": [],
        "trial": [],
        "study": [],
    }
    data_dict = populate_data_dict(data_dirs, data_dict)

    df = pd.DataFrame(data_dict)

    #  check number of participants in each group + distribution of number of files for each
    df = create_unique_ids(df)
    df.groupby(["id", "origin", "label", "study"]).count()["file"].groupby(
        ["origin", "label", "study"]
    ).describe()

    # check length of audio files
    df["duration"] = df["file"].apply(audiofile_duration)
    df.groupby(["origin", "label", "study"]).describe()["duration"]

    # exclude files with duration < 3s
    df[df["duration"] < 3].groupby(["origin", "label"]).count()
    df[df["duration"] < 3].groupby(["id"]).count()["file"].sort_values(ascending=False)
    df[df["duration"] < 3].groupby(["id"]).count()["file"].describe()

    exclude = df[df["duration"] < 3]
    exclude.to_csv("excluded_audio.csv", index=False)

    df = df[~(df["file"].isin(exclude["file"]))]
    # list of ids with audio
    df.groupby(["id"]).count().reset_index()[["id", "file"]].to_csv(
        Path("data") / "multi_diagnosis" / "ids_with_audio.csv", index=False
    )

    # create csvs with train/val/test splits
    train_ids, val_ids, test_ids = read_train_val_test()

    # drop duration column
    df = df.drop("duration", axis=1)

    train_df = df[df["id"].isin(train_ids["ID"])]
    val_df = df[df["id"].isin(val_ids["ID"])]
    test_df = df[df["id"].isin(test_ids["ID"])]

    print(
        f"""{len(train_df)} audio files for training
    {len(val_df)} for validation
    {len(test_df)} for test"""
    )

    if not AUDIO_SPLIT_PATH.exists():
        AUDIO_SPLIT_PATH.mkdir()

    train_df.to_csv(AUDIO_SPLIT_PATH / "audio_train_split.csv", index=False)
    val_df.to_csv(AUDIO_SPLIT_PATH / "audio_val_split.csv", index=False)
    test_df.to_csv(AUDIO_SPLIT_PATH / "audio_test_split.csv", index=False)
