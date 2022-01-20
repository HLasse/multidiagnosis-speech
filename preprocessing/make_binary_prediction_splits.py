"""Creates training files """
import pandas as pd
import numpy as np

from pathlib import Path

from typing import Union


def read_split(filename: str):
    return pd.read_csv(SPLIT_PATH / filename)


def subset_df_by_diagnosis(df: pd.DataFrame, diagnosis: str):
    return df[df["Diagnosis"].isin([diagnosis, "TD"])]


def sample_td_to_diagnosis_ratio(
    df: pd.DataFrame,
    diagnosis: str,
    td_to_diagnosis_ratio: Union[int, float] = 1,
    verbose: bool = True,
):
    """Expects a dataframe with two values in the Diagnosis column (e.g. TD and DEPR).
    Returns a df with all diagnosis rows and td_to_diagnosis ratio as many TD"""
    diagnosis_df = df[df["Diagnosis"] == diagnosis]
    td_df = df[df["Diagnosis"] == "TD"]

    n_diagnosis = len(diagnosis_df)
    td_df = td_df.sample(int(n_diagnosis * td_to_diagnosis_ratio))
    if verbose:
        print(f"{n_diagnosis} from group {diagnosis} matched with {len(td_df)} TD")
    return pd.concat([diagnosis_df, td_df])


def subset_by_id(audio_files_df: pd.DataFrame, id_df: pd.DataFrame):
    df = audio_files_df[audio_files_df["id"].isin(id_df["ID"])]

    print(df.groupby("label")["id"].count())
    return df


if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(42)

    SPLIT_PATH = Path("data") / "splits"
    SAVE_DIR = Path("data") / "audio_file_splits" / "binary_splits"

    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir()

    all_train = read_split("train_split.csv")
    all_val = read_split("validation_split.csv")
    all_test = read_split("test_split.csv")

    all_audio_files = pd.read_csv(
        Path("data") / "multi_diagnosis" / "all_audio_files.csv"
    )

    for diagnosis in ["ASD", "SCHZ", "DEPR"]:
        print(f"[INFO] Processing {diagnosis}...")

        sub_val = subset_df_by_diagnosis(all_val, diagnosis)
        sub_test = subset_df_by_diagnosis(all_test, diagnosis)

        # Strategy 1: keep all TD
        sub_train = subset_df_by_diagnosis(all_train, diagnosis)
        # Strategy 2: even number of TD as diagnosis
        even_train = sample_td_to_diagnosis_ratio(
            sub_train, diagnosis=diagnosis, td_to_diagnosis_ratio=1
        )
        # Strategy 3: 1.5 more TD than diagnosis
        more_td = sample_td_to_diagnosis_ratio(
            sub_train, diagnosis=diagnosis, td_to_diagnosis_ratio=1.5
        )

        # Save different strategies
        filename = f"{diagnosis}_train_split"
        # Subsample all audio files by the ids obtained above and save

        print("[INFO] Number of files...")
        subset_by_id(all_audio_files, sub_train).to_csv(
            SAVE_DIR / (filename + "_all_td.csv"), index=False
        )
        subset_by_id(all_audio_files, even_train).to_csv(
            SAVE_DIR / (filename + "_even_td.csv"), index=False
        )
        subset_by_id(all_audio_files, more_td).to_csv(
            SAVE_DIR / (filename + "_2_to_3_ratio_td.csv"), index=False
        )

        subset_by_id(all_audio_files, sub_val).to_csv(
            SAVE_DIR / f"{diagnosis}_validation_split.csv", index=False
        )
        subset_by_id(all_audio_files, sub_test).to_csv(
            SAVE_DIR / f"{diagnosis}_test_split.csv", index=False
        )
