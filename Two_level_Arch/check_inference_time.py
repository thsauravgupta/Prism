import time
import numpy as np
import torch
import sys
import os

# Ensure the correct path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Two_level_Arch.models.xgboost_model import ContextualReRanker
from Two_level_Arch.models.lstm_model import RoutinePredictor
from Two_level_Arch.models.hybrid_fuser import HybridDecisionEngine
from Two_level_Arch.src.config import LSTM_SEQ_LEN, LSTM_INPUT_SIZE

def measure_time():
    print("Loading models...")
    # Load XGBoost
    xgb_model = ContextualReRanker()
    try:
        xgb_model.load_model()
        num_xgb_features = xgb_model.model.n_features_in_
    except Exception as e:
        print(f"Error loading XGBoost: {e}")
        num_xgb_features = 50 # fallback

    # Load LSTM
    lstm_model = RoutinePredictor(input_size=LSTM_INPUT_SIZE, hidden_size=128, num_layers=2, num_devices=38)
    try:
        lstm_model = RoutinePredictor.load_model()
    except Exception as e:
        print(f"Error loading LSTM: {e}")

    # Initialize fuser
    fuser = HybridDecisionEngine()

    print("Generating dummy data...")
    N = 1000
    num_devices = 38
    
    # Dummy data for XGBoost (N * num_devices, num_features)
    X_xgb = np.random.rand(N * num_devices, num_xgb_features).astype(np.float32)
    
    # Dummy data for LSTM (N, seq_len, input_size)
    X_lstm = np.random.rand(N, LSTM_SEQ_LEN, LSTM_INPUT_SIZE).astype(np.float32)
    X_lstm_tensor = torch.FloatTensor(X_lstm)

    # Dummy heuristic scores (N, num_devices)
    heuristic_scores = np.random.rand(N, num_devices).astype(np.float32)

    print(f"\nEvaluating inference time for batch size N={N} (total {N * num_devices} device evaluations for XGBoost)")
    
    # 1. XGBoost Inference
    start = time.time()
    for _ in range(10):  # Run 10 times for average
        xgb_scores_raw = xgb_model.predict(X_xgb)
        xgb_scores = xgb_scores_raw.reshape(N, num_devices)
    xgb_time = (time.time() - start) / 10
    print(f"XGBoost Inference Time: {xgb_time*1000:.2f} ms per batch of {N}")
    print(f"XGBoost Inference Time: {(xgb_time/N)*1000:.4f} ms per instance")

    # 2. LSTM Inference
    start = time.time()
    lstm_model.eval()
    with torch.no_grad():
        for _ in range(10):
            lstm_logits = lstm_model(X_lstm_tensor)
            lstm_scores = torch.softmax(lstm_logits, dim=-1).numpy()
    lstm_time = (time.time() - start) / 10
    print(f"LSTM Inference Time: {lstm_time*1000:.2f} ms per batch of {N}")
    print(f"LSTM Inference Time: {(lstm_time/N)*1000:.4f} ms per instance")

    # 3. Hybrid Inference (Assuming models have already predicted and we just fuse)
    # But real end-to-end hybrid time is XGBoost + LSTM + Fusion.
    start = time.time()
    for _ in range(10):
        # We assume heuristic is instantaneous or pre-calculated
        fused = fuser.fuse(heuristic_scores, xgb_scores, lstm_scores)
        ranked = fuser.rank(fused, top_k=5)
    fuse_time = (time.time() - start) / 10
    
    total_hybrid_time = xgb_time + lstm_time + fuse_time
    print(f"Fusion only time: {fuse_time*1000:.2f} ms per batch of {N}")
    print(f"Total end-to-end Hybrid Inference Time (LSTM + XGB + Fuser): {total_hybrid_time*1000:.2f} ms per batch of {N}")
    print(f"Total end-to-end Hybrid Inference Time: {(total_hybrid_time/N)*1000:.4f} ms per instance")

if __name__ == "__main__":
    measure_time()
