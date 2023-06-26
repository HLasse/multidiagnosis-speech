"""Identify audio files without transcripts
TODO 
text: look through preprocessing and recreate ids + trial number
audio: check code for making all_audio and add trial id
"""

import pandas as pd

audio_path = "all_audio_files.csv"
text_path = "full_all.csv"

audio = pd.read_csv(audio_path)
text = pd.read_csv(text_path)

audio["id_trial"] = audio["id"].astype(str) + "_" + audio["trial"].astype(str)
audio["has_audio"] = 1
text["id_trial"] = text["ID"] + "_" + text["Trial"].astype(str)
text["has_text"] = 1

audio.groupby(["origin", "label"])["id"].describe()
text.groupby(["Group", "Diagnosis"])["ID"].describe()

audio = audio[["id_trial", "has_audio", "id", "file"]]
text = text[["id_trial", "has_text", "ID"]]

comb = pd.merge(audio, text, how="outer", on="id_trial")
asd = comb[comb["id_trial"].str.startswith("ASD")]


aud_error = asd[asd["has_audio"] != 1]

no_trans = asd[asd["has_text"] != 1]
no_trans["filename"] = no_trans["file"].apply(lambda x: "/".join(x.split("/")[2:]))

no_trans["filename"].to_csv("missing_transcripts.csv", index=False)