import pandas as pd

from pathlib import Path

test_split = pd.read_csv("data/splits/test_split.csv")
audio_split = pd.read_csv("data/audio_file_splits/audio_test_split.csv")
windowed_test = pd.read_csv("data/audio_file_splits/windowed_splits/windowed_test_split.csv")
binary_tst = pd.read_csv("data/audio_file_splits/binary_splits/SCHZ_test_split.csv")

test_split.groupby("Diagnosis").describe()
audio_split.groupby("label")["id"].describe()
windowed_test.groupby("label")["id"].describe()
binary_tst.groupby("label")["id"].describe()

test_set = set(test_split["ID"].unique())
audio_set = set(audio_split["id"].unique())


test_set.intersection(audio_set)
audio_set.union(test_set)

test_set ^ audio_set