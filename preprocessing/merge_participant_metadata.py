"""Updates CleanData with information on which participants have audio and transcriptions.
Fixes a few inconsistencies related to the diagnosis column as well.
creates the "CleanData3.csv" file.
"""
from pathlib import Path

import pandas as pd
import numpy as np


if __name__ == "__main__":

    ## Load data on participants IDs
    transcripts_path = Path().cwd() / "data" / "transcripts" / "full.csv"
    audio_id_path = Path().cwd() / "data" / "multi_diagnosis" / "ids_with_audio.csv"

    ids_with_audio = pd.read_csv(audio_id_path)
    transcripts = pd.read_csv(transcripts_path)

    ids_with_transcript = transcripts["ID"].unique()
    ids_with_audio = ids_with_audio["id"].unique()

    print(
        f"{len(ids_with_audio)} participants with audio.\n{len(ids_with_transcript)} participants with transcripts."
    )

    # Load metadata file
    metadata_path = Path().cwd() / "data" / "multi_diagnosis"

    df = pd.read_csv(metadata_path / "CleanData.csv", sep=";")

    # Save old id
    df["OldID"] = df["ID"]

    ## Convert study for  Diagnosis2 == ChronicDepression to "2"
    df.loc[df["Diagnosis2"] == "ChronicDepression", "Study"] = 2

    # remap names
    df["OverallStudy"] = df["OverallStudy"].map(
        {
            "ASD1": "ASD",
            "ASD2": "ASD",
            "ASD3": "ASD",
            "ASD4": "ASD",
            "Depres1": "DEPR",
            "Schizo1": "SCHZ",
            "Schizo2": "SCHZ",
            "Schizo3": "SCHZ",
            "Schizo4": "SCHZ",
        }
    )

    df["Diagnosis"] = df["Diagnosis"].map(
        {"Control": "TD", "Depression": "DEPR", "Schizophrenia": "SCHZ"}
    )

    ## Add ASD to diagnosis column
    asd_diagnoses = [
        "Aspergers",
        "ASD",
        "F 84.8",
        "F84.5",
        "F84.8",
        "F84.8, F06.2, R41.8",
        "F84.8, R41.8",
        "F84.12",
    ]
    df.loc[
        (df["OverallStudy"] == "ASD") & (df["Diagnosis2"].isin(asd_diagnoses)),
        "Diagnosis",
    ] = "ASD"

    # Construct unique id
    df["ID"] = (
        df["OverallStudy"]
        + "_"
        + df["Diagnosis"]
        + "_"
        + df["Study"].astype(str)
        + "_"
        + df["OldID"].astype(str)
    )

    missing_data = {
        "ID": [
            "SCHZ_SCHZ_2_214",
            "SCHZ_SCHZ_3_330",
            "SCHZ_TD_3_313",
            "SCHZ_TD_4_448",
            "DEPR_DEPR_1_4",
            "DEPR_DEPR_1_44",
            "DEPR_TD_1_42",
        ],
        "Gender": ["Female", "Male", "Male", "Male", "Male", "Female", "Female"],
        "Education": [9, 9, 14, 15, 12, 13, 17],
        "Age": [21, 67, 44, 23, 24, 25, 36],
        "DepressionSeverity": [np.nan, np.nan, np.nan, np.nan, 19, 22, 0],
    }
    missing_data = pd.DataFrame.from_dict(missing_data)
    df = pd.concat([df, missing_data], ignore_index=True)

    # Add indicators
    df["Audio"] = np.where(df["ID"].isin(ids_with_audio), True, False)
    df["Transcription"] = np.where(df["ID"].isin(ids_with_transcript), True, False)

    # Sanity check for duplicates
    ids_occuring_multiple_times = df.groupby("ID").filter(lambda x: len(x) > 1)
    # Two duplicates, but due to error in CleanData. Removing one of them
    ids_occuring_multiple_times = ids_occuring_multiple_times.drop_duplicates()
    # Still one occuring two times (non-important difference)
    duplicate_indices = ids_occuring_multiple_times.index[1:]

    df = df.drop(duplicate_indices)

    df["Gender"] = df["Gender "]
    df = df.drop("Gender ", axis=1)

    df.to_csv(Path().cwd() / "data" / "multi_diagnosis" / "CleanData4.csv", index=False)
    n_ids = df.shape[0]

    print(
        f"{n_ids} participants in total.\n{(n_ids - df['Audio'].sum())} without audio.\n{(n_ids - df['Transcription'].sum())} without transcription."
    )

    df_both_audio_and_transcripts = df[
        (df["Audio"] == True) & (df["Transcription"] == True)
    ]
    print(f"{df_both_audio_and_transcripts.shape[0]} with both audio and transcription")
