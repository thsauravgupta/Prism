"""
config.py — Central configuration for the SmartThings Prediction Pipeline.

Contains all hyperparameters, paths, device mappings, and constants used
across the data pipeline, models, and evaluation.
"""

import os
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")          # extracted SmartSense data
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


COUNTRIES = ["kr", "us", "sp", "fr"]
DEFAULT_COUNTRY = "kr"   


ACTION_FEATURES = ["day_of_week", "hour", "device", "control", "device_control"]
SEQUENCE_LENGTH = 10   # each instance = 10 consecutive actions


DEFAULT_DEVICE_TYPES = {
    0: "TV",
    1: "AC",
    2: "Light",
    3: "Washer",
    4: "Oven",
    5: "Fridge",
    6: "Speaker",
    7: "Vacuum",
    8: "Plug",
    9: "Thermostat",
}


HEURISTIC_W1 = 0.4   
HEURISTIC_W2 = 0.4   
HEURISTIC_W3 = 0.2   


XGB_PARAMS = {
    "objective": "rank:pairwise",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 1,
}


LSTM_INPUT_SIZE = 5         # 5 features per action in SmartSense
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = False

# Training hyperparameters
LSTM_LR = 1e-3
LSTM_WEIGHT_DECAY = 1e-5
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 256
LSTM_SEQ_LEN = 9       # first 9 steps as input → predict 10th step target
LSTM_PATIENCE = 7       # early stopping patience


FUSION_WEIGHTS = {
    "heuristic": 0.3,
    "xgboost": 0.35,
    "lstm": 0.35,
}
TOP_K = 5  # return top-K predictions


RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2


EVAL_K_VALUES = [1, 3, 5, 10]
