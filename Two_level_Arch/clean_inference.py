import time
import numpy as np
import torch
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Two_level_Arch.models.xgboost_model import ContextualReRanker
from Two_level_Arch.models.lstm_model import RoutinePredictor
from Two_level_Arch.models.hybrid_fuser import HybridDecisionEngine
from Two_level_Arch.src.config import LSTM_SEQ_LEN, LSTM_INPUT_SIZE

def measure():
    xgb_model = ContextualReRanker()
    xgb_model.load_model()
    num_xgb_features = xgb_model.model.n_features_in_
    lstm_model = RoutinePredictor.load_model()
    fuser = HybridDecisionEngine()
    
    N = 1000
    num_devices = 38
    X_xgb = np.random.rand(N * num_devices, num_xgb_features).astype(np.float32)
    X_lstm_tensor = torch.FloatTensor(np.random.rand(N, LSTM_SEQ_LEN, LSTM_INPUT_SIZE).astype(np.float32))
    heuristic_scores = np.random.rand(N, num_devices).astype(np.float32)

    # xgb
    start = time.time()
    for _ in range(5):
        xgb_scores = xgb_model.predict(X_xgb).reshape(N, num_devices)
    xgb_time = (time.time() - start) / 5
    
    # lstm
    start = time.time()
    lstm_model.eval()
    with torch.no_grad():
        for _ in range(5):
            lstm_scores = torch.softmax(lstm_model(X_lstm_tensor), dim=-1).numpy()
    lstm_time = (time.time() - start) / 5
    
    # fuser
    start = time.time()
    for _ in range(5):
        ranked = fuser.rank(fuser.fuse(heuristic_scores, xgb_scores, lstm_scores), 5)
    fuse_time = (time.time() - start) / 5

    res = {
        "N": N,
        "xgb_ms_total": xgb_time * 1000,
        "xgb_ms_per_instance": (xgb_time / N) * 1000,
        "lstm_ms_total": lstm_time * 1000,
        "lstm_ms_per_instance": (lstm_time / N) * 1000,
        "hybrid_ms_total": (xgb_time + lstm_time + fuse_time) * 1000,
        "hybrid_ms_per_instance": ((xgb_time + lstm_time + fuse_time) / N) * 1000,
    }
    with open("inference_results.json", "w") as f:
        json.dump(res, f)

if __name__ == "__main__":
    measure()
