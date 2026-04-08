

import os
import pickle
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

from .config import (
    DATA_DIR, COUNTRIES, DEFAULT_COUNTRY,
    ACTION_FEATURES, SEQUENCE_LENGTH,
)


SMARTSENSE_URL = "https://github.com/snudatalab/SmartSense/raw/main/data.zip"


def download_smartsense(dest_dir: str = DATA_DIR, force: bool = False) -> str:
    
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "data.zip")


    if not force and os.path.isdir(os.path.join(dest_dir, "kr")):
        print(f"[INFO] SmartSense data already exists at {dest_dir}")
        return dest_dir

    if not os.path.isfile(zip_path) or force:
        print(f"[INFO] Downloading SmartSense data from {SMARTSENSE_URL} ...")
        urllib.request.urlretrieve(SMARTSENSE_URL, zip_path)
        print(f"[INFO] Downloaded to {zip_path}")

  
    print(f"[INFO] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"[INFO] Extracted to {dest_dir}")

    return dest_dir



def _load_pkl(filepath: str) -> np.ndarray:
    """Load a pickle file and return as numpy array."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    if hasattr(data, "numpy"):  # torch tensor
        return data.numpy()
    return np.array(data)


def load_country_data(
    country: str = DEFAULT_COUNTRY,
    data_dir: str = DATA_DIR,
) -> Dict[str, np.ndarray]:
    
    country_dir = os.path.join(data_dir, country)
    if not os.path.isdir(country_dir):
        raise FileNotFoundError(
            f"Country data '{country}' not found at {country_dir}. "
            f"Run download_smartsense() first."
        )

    splits = {}
    split_files = {
        "train": "trn_instance_10.pkl",
        "val": "vld_instance_10.pkl",
        "test": "test_instance_10.pkl",
    }

    for split_name, filename in split_files.items():
        filepath = os.path.join(country_dir, filename)
        if os.path.isfile(filepath):
            splits[split_name] = _load_pkl(filepath)
            print(f"[INFO] Loaded {split_name}: {splits[split_name].shape}")
        else:
            print(f"[WARN] {filename} not found in {country_dir}")

    return splits


def load_dictionary(
    country: str = DEFAULT_COUNTRY,
    data_dir: str = DATA_DIR,
) -> Optional[Dict]:
    """Load the dictionary.py mappings for a country.

    Returns a dict with keys like 'device', 'control', 'day_of_week', etc.
    mapping names to IDs.
    """
    dict_path = os.path.join(data_dir, country, "dictionary.py")
    if not os.path.isfile(dict_path):
        print(f"[WARN] dictionary.py not found for country '{country}'")
        return None

    
    d = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        exec(f.read(), d)

    
    mappings = {k: v for k, v in d.items() if isinstance(v, dict) and not k.startswith("_")}
    print(f"[INFO] Loaded dictionary for '{country}': {list(mappings.keys())}")
    return mappings


def load_routines(
    country: str = DEFAULT_COUNTRY,
    data_dir: str = DATA_DIR,
) -> Optional[List[List[int]]]:
    
    routine_path = os.path.join(data_dir, country, "routine_device_corpus.txt")
    if not os.path.isfile(routine_path):
        print(f"[WARN] routine_device_corpus.txt not found for '{country}'")
        return None

    routines = []
    with open(routine_path, "r") as f:
        for line in f:
            devices = [int(x) for x in line.strip().split() if x]
            if devices:
                routines.append(devices)

    print(f"[INFO] Loaded {len(routines)} routines for '{country}'")
    return routines



def instances_to_dataframe(instances: np.ndarray) -> pd.DataFrame:
    
    N, seq_len, feat_dim = instances.shape
    rows = []
    for i in range(N):
        for t in range(seq_len):
            row = {
                "instance_id": i,
                "step": t,
            }
            for f_idx, f_name in enumerate(ACTION_FEATURES):
                row[f_name] = int(instances[i, t, f_idx])
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def load_all_countries(data_dir: str = DATA_DIR) -> Dict[str, Dict[str, np.ndarray]]:
    
    all_data = {}
    for country in COUNTRIES:
        country_dir = os.path.join(data_dir, country)
        if os.path.isdir(country_dir):
            all_data[country] = load_country_data(country, data_dir)
    return all_data
