"""Generate train/test split"""
from collections import defaultdict
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import random
import re

BOTH_GENDERS = False
SAVE_IDS = True
ID_SAVE_PATH = os.path.join("data", "id_train_test_split_males.csv")
IDS_FROM_CSV = os.path.join("data", "id_train_test_split_males.csv")

def get_id(filename):
    name = filename.split("_")[-2]
    if name.endswith("A"):
        return name[:-1]
    if len(name.split(".")) > 1:
        return name.split(".")[0]
    pattern = re.compile("^[A-Z]{3}")
    regex_match = pattern.search(name)
    if regex_match:
        return regex_match[0]
    return name

def get_task(filename):
    return filename.split("_")[1]

def get_trial(filename):
    return filename.split("_")[-1][:-4]

def sample_ids(df, diagnosis, gender, k):
    return random.choices(df[(df["Diagnosis"] == diagnosis) & (df["Gender"] == gender)]["ID"].tolist()[0], k=k)

def train_test_split_to_csv(test_ids, train_ids):
    test_ids = {"ID" : test_ids,
                "split" : "test"}
    train_ids = {"ID" : train_ids,
                "split" : "train"}
    test_ids = pd.DataFrame(test_ids)
    train_ids = pd.DataFrame(train_ids)
    id_split = test_ids.append(train_ids)
    id_split.to_csv(ID_SAVE_PATH, index=False)

def get_train_test_from_csv():
        train_test_df = pd.read_csv(IDS_FROM_CSV)
        train_ids = train_test_df[train_test_df["split"] == "train"]["ID"].astype(str).tolist()
        test_ids = train_test_df[train_test_df["split"] == "test"]["ID"].astype(str).tolist()
        return train_ids, test_ids

if __name__ == "__main__":
    data = os.path.join(os.getcwd(), "data", "autism_data")
    key = os.path.join(os.getcwd(), "data", "autism_data_key", "FullDataByTrial.csv")
    filepaths = os.listdir(data)
    file_dict = [{"file": os.path.join(data, path)} for path in filepaths if path.endswith(".wav")]

    df_key = pd.read_csv(key)
    df = pd.DataFrame(file_dict)
    df["name"] = df.file.apply(lambda x: x.split("/")[-1])
    df["ID"] = df.name.apply(lambda x: get_id(x))
    unique_wav_ids = set(df["ID"])
    unique_key_ids = set(df_key["ID"])

    diff_ids = unique_wav_ids - unique_key_ids
    # print(diff_ids)

    # make new key rows for the known missing 
    key_cols = df_key.columns
    new_rows = defaultdict()
    for col in key_cols:
        new_rows[col] = [np.nan] * 4
    new_rows["ID"] = ["7320", "7231", "7029", "7094"]
    new_rows["Diagnosis"] = ["ASD"] * 4
    new_rows["Language"] = ["us"] * 4
    new_rows = pd.DataFrame(new_rows)
    df_key = df_key.append(new_rows)

    keep_cols = [
        "ID", 
        "Language",
        "Diagnosis",
        "Gender",
        "Age",
        "AdosCommunication",
        "AdosSocial",
        "AdosCreativity",
        "AdosStereotyped",
        "VIQ",
        "PIQ",
        "TIQ",
        "ParentalEducation",
        "SRS",
        "CARS",
        "PPVT",
        "Leiter",
        "language",
        "AgeS"]
    df_merge_key = df_key[keep_cols]
    df_merge_key = df_merge_key.drop_duplicates()
    df = df.merge(df_merge_key, on="ID")

    df["Trial"] = df.name.apply(lambda x: get_trial(x))
    df["Task"] = df.name.apply(lambda x: get_task(x))

    # only keeping the Danish data
    df = df[df["language"] == "dk"]

    if BOTH_GENDERS:
    # stratifying - sampling 2 females and 4 males from each group
        unique_ids = pd.DataFrame(df.groupby(["Diagnosis", "Gender"])["ID"].unique()).reset_index()
        asd_male = sample_ids(unique_ids, "ASD", "Male", k=4)
        asd_female = sample_ids(unique_ids, "ASD", "Female", k=2)
        td_male = sample_ids(unique_ids, "TD", "Male", k=4)
        td_female = sample_ids(unique_ids, "TD", "Female", k=2)

        test_ids = asd_male + asd_female + td_male + td_female
        all_ids = np.concatenate(unique_ids["ID"].tolist())

    else:
        df = df[df["Gender"] == "Male"]
        test_ids = ["109", "121", "126", "127", "135", "211", "212", "219", "228"]
        all_ids = df["ID"].unique()
    
    train_ids = [x for x in list(all_ids) if x not in test_ids]

    if SAVE_IDS:
        train_test_split_to_csv(test_ids, train_ids)

    if IDS_FROM_CSV:
        train_ids, test_ids = get_train_test_from_csv()

    train = df[df["ID"].isin(train_ids)]
    test = df[df["ID"].isin(test_ids)]

    stories_train = train[train["Task"] == "stories"]
    stories_test = test[test["Task"] == "stories"]
    triangles_train = train[train["Task"] == "triangles"]
    triangles_test = test[test["Task"] == "triangles"]

    stories_train.to_csv(os.path.join("data", "splits", f"stories_train_data_gender_{str(BOTH_GENDERS)}.csv"), index=False)
    stories_test.to_csv(os.path.join("data", "splits", f"stories_test_data_gender_{str(BOTH_GENDERS)}.csv"), index=False)
    triangles_train.to_csv(os.path.join("data", "splits", f"triangles_train_data_gender_{str(BOTH_GENDERS)}.csv"), index=False)
    triangles_test.to_csv(os.path.join("data", "splits", f"triangles_test_data_gender_{str(BOTH_GENDERS)}.csv"), index=False)




