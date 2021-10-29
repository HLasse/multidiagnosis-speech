"""Generate train/test split"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
import re

BOTH_GENDERS = False
SAVE_IDS = True
ID_SAVE_PATH = os.path.join("data", "id_train_test_split_males.csv")
IDS_FROM_CSV = os.path.join("data", "id_train_test_split_males.csv")


def get_metadata(filename):
    """Get id and trial from filename"""
    file = Path(filename)
    group = file.parent.parent.parent.name
    if group in ["ASD", "SCHZ"]:
        id = re.findall("S(\d\d\d)", file.name)
        trial = re.findall("T(\d)", file.name)
    if group == "DEPR":
        id = re.findall("d[pc](\d+)", file.name)
        trial = re.findall("_(\d+)", file.name)
    if not id:
        ValueError(f"No ID found in filename {file}")
    if not trial:
        ValueError(f"No trial found in filename {file}")
    return (id[0], trial[0])


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
            files = get_files_from_dir(diagnosis / "TD")
            data_dict = update_data_dict(data_dict, files, "TD")
        if diagnosis.name == "SCHZ":
            files = get_files_from_dir(diagnosis / "schizophrenic")
            data_dict = update_data_dict(data_dict, files, "SCHZ")
            files = get_files_from_dir(diagnosis / "TD")
            data_dict = update_data_dict(data_dict, files, "TD")
    # flatten
    data_dict = {k : flatten(v) for k, v in data_dict.items()}
    return data_dict

def update_data_dict(data_dict, files, label):
    """update metadata dictionary based on information in filename"""
    data_dict["file"].append(files)
    data_dict["label"].append([label] * len(files))
    data_dict["origin"].append([files[0].split("/")[-4]] * len(files))
    ids, trials = zip(*[get_metadata(f) for f in files])
    data_dict["id"].append(ids)
    data_dict["trial"].append(trials)
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


if __name__ == "__main__":
    data_dirs = os.path.join(os.getcwd(), "data", "multi_diagnosis")
    data_dict = {"file" : [], "label" : [], "origin" : [], "id" : [], "trial" : []}
    data_dict = populate_data_dict(data_dirs, data_dict)
    
    df = pd.DataFrame(data_dict)

    #  check number of participants in each group + distribution of number of files for each
    df.groupby(["origin", "label", "id"]).count()["file"].groupby(["origin", "label"]).describe()

    ### TODO
    # make train/val/test split with Roberta and Riccardo or set up IDS for CV
    # 


