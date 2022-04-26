"""Creates the training/validation/test split.
Test split contains 6 IDs + their matching controls from each diagnostic group, balanced by gender, that both have audio and transcriptions.
Validation split contains 6 IDs from each diagnostic group, balanced by gender.
Only few ASD in the validation set have transcriptions, (as only few ASD have transcriptions).
The splits needs to be processed by ´preproc_audio.py´ before they can be input to ´train_wav2vec.py´

"""

import pandas as pd
from pathlib import Path
import numpy as np



def sample_even_distribution(df_in: pd.DataFrame, metadata_df: pd.DataFrame, n_samples=3):
    df_test = df_in.groupby(["Diagnosis", "Gender"]).sample(n_samples)
    # Get matching controls
    test_matching_controls = [get_matching_control(id) for id in df_test["ID"]]
    controls = df[df["ID"].isin(test_matching_controls)]

    if df_test.shape[0] == controls.shape[0]:
        return pd.concat([df_test, controls])
    else:
        return sample_even_distribution(df_in, metadata_df, n_samples)



def get_matching_control(id):
    split_id = id.split("_")
    split_id[1] = "TD"
    ## if id = ASD and study 1, add 100 to get the matching control
    if split_id[0] == "ASD" and split_id[2] == "1":
        split_id[3] = str(int(split_id[3]) + 100) 
    ## Same TD is matched to both chronic and 1ep depression    
    if split_id[0] == "DEPR" and split_id[2] == "2":
        split_id[2] = "1"
    control_id = "_".join(split_id)
    return control_id

if __name__ == "__main__":

    # pretty print df
    pd.set_option("expand_frame_repr", False)

    # set seed for reproducible splits
    np.random.seed(42)

    # Load metadata
    metadata_path = Path().cwd() / "data" / "multi_diagnosis" / "CleanData4.csv"
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

    ## Remove TDs from dataset before sampling
    df_both_audio_and_transcripts_no_td = df_both_audio_and_transcripts[df_both_audio_and_transcripts["Diagnosis"] != "TD"]
    # remove the F/M gender to be able to stratify sampling
    df_both_audio_and_transcripts_no_td = df_both_audio_and_transcripts_no_td[df_both_audio_and_transcripts_no_td["Gender"] != "F/M"]

    # sample 3 from each group diagnostic group + matching controls for test set
    df_test = sample_even_distribution(df_both_audio_and_transcripts_no_td, df)
    
    # remove test set from remaining data
    df_no_test = df.drop(df_test.index)
    df_both_audio_and_transcripts_no_td_no_test = df_both_audio_and_transcripts_no_td.drop(df_test[df_test["Diagnosis"] != "TD"].index)
    
    # sample for validation set
    df_val = sample_even_distribution(df_both_audio_and_transcripts_no_td_no_test, df_no_test)

    # train set is the remainder
    df_train = df_no_test.drop(df_val.index)

    ## save splits
    save_dir = Path("data") / "splits"
    if not save_dir.exists():
        save_dir.mkdir()

    df_test.to_csv(save_dir / "test_split.csv", index=False)
    df_val.to_csv(save_dir / "validation_split.csv", index=False)
    df_train.to_csv(save_dir / "train_split.csv", index=False)

    print(f"{df_train.shape[0]} participants in train set")

    print("Diagnosis and gender distribution of train set")
    print(df_train.groupby(["Diagnosis", "Gender"])["ID"].describe()["count"])

    print("Diagnosis and gender distribution of validation set")
    print(df_val.groupby(["Diagnosis", "Gender"])["ID"].describe()["count"])

    print("Diagnosis and gender distribution of test set")
    print(df_test.groupby(["Diagnosis", "Gender"])["ID"].describe()["count"])


    # very few ASD with transcripts!
    # very few males with depression
