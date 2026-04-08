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
CONTEXT_LENGTH = 9     # first 9 steps used as context, 10th is prediction target
NUM_HOUR_BINS = 8      # SmartSense hours are binned into 8 slots (0-7)
NUM_DAY_BINS = 7       # 7 days of the week (0-6)


# Real SmartSense device types (from dictionary.py — Korea dataset)
DEFAULT_DEVICE_TYPES = {
    0: "AirConditioner", 1: "AirPurifier", 2: "Blind", 3: "Camera",
    4: "ClothingCareMachine", 5: "Computer", 6: "ContactSensor",
    7: "Dishwasher", 8: "DoorBell", 9: "Dryer", 10: "Elevator",
    11: "Fan", 12: "GarageDoor", 13: "GasValve", 14: "Humidifier",
    15: "LeakSensor", 16: "Light", 17: "Microwave",
    18: "MotionSensor", 19: "MultiFunctionalSensor",
    20: "NetworkAudio", 21: "None", 22: "Other",
    23: "PresenceSensor", 24: "Projector", 25: "Refrigerator",
    26: "RemoteController", 27: "RobotCleaner", 28: "SetTop",
    29: "Siren", 30: "SmartLock", 31: "SmartPlug", 32: "Switch",
    33: "Television", 34: "Thermostat", 35: "Vent",
    36: "Washer", 37: "WaterValve",
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
LSTM_SEQ_LEN = CONTEXT_LENGTH  # first 9 steps as input → predict 10th step target
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
