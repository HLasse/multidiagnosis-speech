from pathlib import Path

import pandas as pd
import numpy as np

data_path = Path().cwd() / "data" / "multi_diagnosis"

# load data
df = pd.read_csv(data_path / "CleanData2.csv", sep=";")

df["Audio"] = np.where(df["Audio"] == 1.0, True, False)
df["Transcription"] = np.where(df["Transcription"] == 1.0, True, False)

n_ids = df.shape[0]

print(f"{n_ids} participants in total.\n{(n_ids - df['Audio'].sum())} without audio.\n{(n_ids - df['Transcription'].sum())} without transcription.")

# identify test set (6 from each group, 3/3 from 1st episode dep/chronic, 3 females, 3 males)
df_both_audio_and_transcripts = df[(df["Audio"] == True) & (df["Transcription"] == True)]
print(f"{df_both_audio_and_transcripts.shape[0]} with both audio and transcription")

## Convert study for  Diagnosis2 == ChronicDepression to "2"
for col in ["Study", "OldID"]:
    df[col] = df[col].astype(int)



ids_occuring_multiple_times = df["ID"].value_counts() > 1

ids_occuring_multiple_times = df.groupby("ID").filter(lambda x: len(x) > 1)

# check the nan diagnoses and recode them properly




df["Diagnosis"].unique()

df[df["Diagnosis"].isna()]

# make sure test set has both audio and transcription

# randomly split remainder into train and val

