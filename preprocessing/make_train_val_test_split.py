"""Creates the training/validation/test split.
Test split contains 6 IDs from each diagnostic group, balanced by gender, that both have audio and transcriptions.
Validation split contains 6 IDs from each diagnostic group, balanced by gender.
Only few ASD in the validation set have transcriptions, (as only few ASD have transcriptions).
"""

import pandas as pd
from pathlib import Path
import numpy as np

if __name__ == "__main__":

    # pretty print df
    pd.set_option("expand_frame_repr", False)

    # set seed for reproducible splits
    np.random.seed(42)

    # Load metadata
    metadata_path = Path().cwd() / "data" / "multi_diagnosis" / "CleanData3.csv"
    df = pd.read_csv(metadata_path)

    n_ids = df.shape[0]

    print(
        f"{n_ids} participants in total.\n{(n_ids - df['Audio'].sum())} without audio.\n{(n_ids - df['Transcription'].sum())} without transcription."
    )

    df_both_audio_and_transcripts = df[
        (df["Audio"] == True) & (df["Transcription"] == True)
    ]

    ### identify test set
    # (6 from each group, 3/3 from 1st episode dep/chronic, 3 females, 3 males)

    print("Diagnosis / gender distribution for all participants:")
    print(df.groupby(["Diagnosis", "Gender"])["ID"].describe())

    print(
        "Diagnosis / gender distribution for participants with both audio and transcripts:"
    )
    print(df_both_audio_and_transcripts.groupby(["Diagnosis", "Gender"])["ID"].describe())

    df_test = df_both_audio_and_transcripts.groupby(["Diagnosis", "Gender"]).sample(3)

    df_no_test = df.drop(df_test.index)

    # remove the F/M gender to be able to stratify sampling
    df_no_test_no_fm_gender = df_no_test[~(df_no_test["Gender"] == "F/M")]

    df_val = df_no_test_no_fm_gender.groupby(["Diagnosis", "Gender"]).sample(3)

    df_train = df_no_test.drop(df_val.index)

    ## save splits
    save_dir = Path("data") / "splits"
    if not save_dir.exists():
        save_dir.mkdir()

    df_test.to_csv(save_dir / "test_split.csv", index=False)
    df_val.to_csv(save_dir / "validation_split.csv", index=False)
    df_train.to_csv(save_dir / "train_split.csv", index=False)


    print("Test IDs:")
    print(
        df_test[
            ["ID", "Gender", "Diagnosis", "Diagnosis2", "Age", "Audio", "Transcription"]
        ]
    )

    print("Validation IDs")
    df_val[["ID", "Gender", "Diagnosis", "Diagnosis2", "Age", "Audio", "Transcription"]]

    print(f"{df_train.shape[0]} participants in train set")

    print("Diagnosis and gender distribution of train set")
    print(df_train.groupby(["Diagnosis", "Gender"])["ID"].describe()["count"])


    # very few ASD with transcripts!
    # very few males with depression
