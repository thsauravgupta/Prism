#!/usr/bin/env python3
"""
demo_inference.py — Standalone Inference Demo Script
=====================================================

SmartThings On-Device Predictive Model Integration

This script demonstrates end-to-end inference for both predictive pipelines:
  1. Two-Level Architecture  (Heuristic + XGBoost + LSTM → Hybrid Fuser)
  2. KG-GNN                  (Knowledge Graph GNN via TensorFlow Lite)

It loads the pre-trained models, generates sample input data matching the
SmartSense dataset format, runs predictions, and prints results.

Usage:
    python demo_inference.py --all          # Run both pipelines
    python demo_inference.py --two-level    # Run Two-Level Architecture only
    python demo_inference.py --kg-gnn       # Run KG-GNN only
    python demo_inference.py --benchmark    # Run with latency benchmarks

Requirements:
    pip install -r requirements-inference.txt

Can also run inside Docker:
    docker compose up --build
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Fix Windows encoding issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Constants ───────────────────────────────────────────────────────────────
DEVICE_TYPES = {
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

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def print_header(title: str):
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_predictions(predictions: list, title: str = "Top-5 Predicted Devices"):
    """Pretty-print a list of (rank, device_name, score) predictions."""
    print(f"\n  {title}")
    print(f"  {'Rank':<6} {'Device':<25} {'Score':>10}")
    print(f"  {'-'*6} {'-'*25} {'-'*10}")
    for rank, (dev_id, dev_name, score) in enumerate(predictions, 1):
        print(f"  #{rank:<5} {dev_name:<25} {score:>10.4f}")


def generate_sample_sequence(seed: int = 42) -> np.ndarray:
    """Generate a realistic sample sequence of 10 SmartThings actions.

    Simulates a morning routine:
      - Wake up (MotionSensor), turn on Light, check Television,
        open Blind, use AirConditioner, check Refrigerator, etc.

    Returns:
        Array of shape [10, 5] with columns:
        [day_of_week, hour, device, control, device_control]
    """
    np.random.seed(seed)

    # Simulate a plausible morning routine on a Monday (day=0)
    routine = [
        # day, hour_bin, device_id, control_id, device_control_id
        [0, 2, 18, 1, 30],   # 06:00 – MotionSensor detects movement
        [0, 2, 16, 1, 25],   # 06:05 – Light turned on
        [0, 2, 33, 1, 50],   # 06:10 – Television turned on
        [0, 3, 2,  1, 5],    # 07:00 – Blind opened
        [0, 3, 0,  2, 2],    # 07:15 – AirConditioner set to cool
        [0, 3, 25, 1, 40],   # 07:30 – Refrigerator door opened
        [0, 4, 16, 0, 24],   # 08:00 – Light turned off
        [0, 4, 36, 1, 55],   # 08:15 – Washer started
        [0, 4, 30, 1, 45],   # 08:30 – SmartLock locked
        # Target: what device will user interact with next?
        [0, 5, 33, 0, 49],   # 09:00 – Television turned off (this is the target)
    ]
    return np.array(routine, dtype=np.float32)


def generate_batch_sequences(n: int = 100, seed: int = 42) -> np.ndarray:
    """Generate a batch of random sequences for benchmarking.

    Returns:
        Array of shape [N, 10, 5]
    """
    np.random.seed(seed)
    data = np.zeros((n, 10, 5), dtype=np.float32)
    data[:, :, 0] = np.random.randint(0, 7, size=(n, 10))     # day_of_week
    data[:, :, 1] = np.random.randint(0, 8, size=(n, 10))     # hour_bin
    data[:, :, 2] = np.random.randint(0, 38, size=(n, 10))    # device
    data[:, :, 3] = np.random.randint(0, 15, size=(n, 10))    # control
    data[:, :, 4] = np.random.randint(0, 60, size=(n, 10))    # device_control
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 1: Two-Level Architecture
# ═══════════════════════════════════════════════════════════════════════════

def run_two_level_inference(benchmark: bool = False):
    """Run the Two-Level Architecture inference pipeline.

    Components loaded:
      - HeuristicScorer:       Rule-based (recency + frequency + power)
      - ContextualReRanker:    XGBoost learning-to-rank (rank:pairwise)
      - RoutinePredictor:      2-layer LSTM sequence model
      - HybridDecisionEngine:  Weighted fusion of all three
    """
    import torch
    from Two_level_Arch.models.heuristic import HeuristicScorer
    from Two_level_Arch.models.xgboost_model import ContextualReRanker
    from Two_level_Arch.models.lstm_model import RoutinePredictor
    from Two_level_Arch.models.hybrid_fuser import HybridDecisionEngine
    from Two_level_Arch.src.feature_engineering import (
        compute_recency, compute_frequency, compute_power_proxy,
    )
    from Two_level_Arch.src.config import FUSION_WEIGHTS

    print_header("Pipeline 1: Two-Level Architecture Inference")

    num_devices = 38  # SmartSense device types

    # ── Step 1: Load models ─────────────────────────────────────────────
    print("\n  [1/5] Loading pre-trained models...")

    heuristic = HeuristicScorer(w1=0.4, w2=0.4, w3=0.2)
    print(f"         ✓ HeuristicScorer loaded (weights: {heuristic.get_weights()})")

    xgb_model = ContextualReRanker()
    xgb_model.load_model()
    n_features = xgb_model.model.n_features_in_
    print(f"         ✓ XGBoost ContextualReRanker loaded ({n_features} features)")

    lstm_model = RoutinePredictor.load_model()
    lstm_model.eval()
    print(f"         ✓ LSTM RoutinePredictor loaded "
          f"(hidden={lstm_model.hidden_size}, layers={lstm_model.num_layers})")

    fuser = HybridDecisionEngine(weights=FUSION_WEIGHTS)
    print(f"         ✓ HybridDecisionEngine loaded (weights: {FUSION_WEIGHTS})")

    # ── Step 2: Prepare sample input ────────────────────────────────────
    print("\n  [2/5] Preparing sample input...")
    sample = generate_sample_sequence()
    context = sample[:-1]  # First 9 steps (input)
    target_device = int(sample[-1, 2])  # 10th step device = ground truth
    print(f"         Input: {context.shape[0]} action steps, 5 features each")
    print(f"         Ground truth target: {DEVICE_TYPES.get(target_device, '?')} "
          f"(id={target_device})")

    # ── Step 3: Compute heuristic scores ────────────────────────────────
    print("\n  [3/5] Computing heuristic scores...")
    recency = compute_recency(context, num_devices)
    frequency = compute_frequency(context, num_devices)
    power = compute_power_proxy(context, num_devices)
    heuristic_scores = heuristic.score_batch(recency, frequency, power)
    print(f"         Heuristic scores shape: {heuristic_scores.shape}")

    heur_top5 = np.argsort(heuristic_scores)[::-1][:5]
    print_predictions(
        [(int(d), DEVICE_TYPES.get(int(d), f"device_{d}"), heuristic_scores[d])
         for d in heur_top5],
        title="Heuristic Top-5"
    )

    # ── Step 4: Run XGBoost re-ranking ─────────────────────────────────
    print("\n  [4/5] Running XGBoost re-ranking...")

    # Build feature vectors for each candidate device
    hours = context[:, 1].astype(int)
    days = context[:, 0].astype(int)
    avg_hour = np.mean(hours) / 7.0
    day_dist = np.zeros(7)
    for d in days:
        if d < 7:
            day_dist[d] += 1
    day_dist = day_dist / max(day_dist.sum(), 1)

    xgb_features = []
    for dev_id in range(num_devices):
        feat = np.concatenate([
            [recency[dev_id]],
            [frequency[dev_id]],
            [power[dev_id]],
            [avg_hour],
            day_dist,
        ])
        xgb_features.append(feat)
    X_xgb = np.array(xgb_features, dtype=np.float32)
    xgb_scores = xgb_model.predict(X_xgb)
    print(f"         XGBoost scores shape: {xgb_scores.shape}")

    xgb_top5 = np.argsort(xgb_scores)[::-1][:5]
    print_predictions(
        [(int(d), DEVICE_TYPES.get(int(d), f"device_{d}"), xgb_scores[d])
         for d in xgb_top5],
        title="XGBoost Top-5"
    )

    # ── Step 5: Run LSTM prediction ────────────────────────────────────
    print("\n  [5/5] Running LSTM routine prediction...")

    # Normalise input for LSTM (matching feature_engineering.py)
    X_lstm = context.copy().astype(np.float32)
    X_lstm[:, 0] = X_lstm[:, 0] / 6.0   # day
    X_lstm[:, 1] = X_lstm[:, 1] / 7.0   # hour
    max_device = max(X_lstm[:, 2].max(), 1)
    max_control = max(X_lstm[:, 3].max(), 1)
    max_dc = max(X_lstm[:, 4].max(), 1)
    X_lstm[:, 2] /= max_device
    X_lstm[:, 3] /= max_control
    X_lstm[:, 4] /= max_dc

    X_tensor = torch.FloatTensor(X_lstm).unsqueeze(0)  # [1, 9, 5]
    with torch.no_grad():
        logits = lstm_model(X_tensor)
        lstm_probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    print(f"         LSTM probabilities shape: {lstm_probs.shape}")

    lstm_top5 = np.argsort(lstm_probs)[::-1][:5]
    print_predictions(
        [(int(d), DEVICE_TYPES.get(int(d), f"device_{d}"), lstm_probs[d])
         for d in lstm_top5],
        title="LSTM Top-5"
    )

    # ── Fusion ─────────────────────────────────────────────────────────
    print_header("Hybrid Fusion — Final Results")

    heuristic_2d = heuristic_scores.reshape(1, -1)
    xgb_2d = xgb_scores.reshape(1, -1)
    lstm_2d = lstm_probs.reshape(1, -1)

    ranked_indices, fused_scores = fuser.predict(
        heuristic_2d, xgb_2d, lstm_2d, top_k=5
    )
    ranked = ranked_indices[0]
    scores = fused_scores[0]

    print(f"\n  Fusion weights: heuristic={FUSION_WEIGHTS['heuristic']}, "
          f"xgboost={FUSION_WEIGHTS['xgboost']}, lstm={FUSION_WEIGHTS['lstm']}")

    final_predictions = [
        (int(d), DEVICE_TYPES.get(int(d), f"device_{d}"), float(scores[d]))
        for d in ranked
    ]
    print_predictions(final_predictions, title="★ Fused Top-5 Predictions")

    hit = target_device in ranked
    print(f"\n  Ground truth: {DEVICE_TYPES.get(target_device)} (id={target_device})")
    print(f"  Hit@5: {'✅ YES — target is in top-5' if hit else '❌ NO — target not in top-5'}")

    # ── Benchmark (optional) ───────────────────────────────────────────
    if benchmark:
        print_header("Inference Benchmark — Two-Level Architecture")
        batch = generate_batch_sequences(n=1000)
        _benchmark_two_level(batch, heuristic, xgb_model, lstm_model, fuser,
                             num_devices, n_features)

    # ── Save results ───────────────────────────────────────────────────
    result = {
        "pipeline": "Two-Level Architecture",
        "ground_truth": {"device_id": target_device,
                         "device_name": DEVICE_TYPES.get(target_device)},
        "predictions": [{"rank": i + 1, "device_id": int(d),
                         "device_name": DEVICE_TYPES.get(int(d)),
                         "fused_score": float(scores[d])}
                        for i, d in enumerate(ranked)],
        "hit_at_5": bool(hit),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "two_level_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to output/two_level_results.json")

    return result


def _benchmark_two_level(data, heuristic, xgb_model, lstm_model, fuser,
                         num_devices, n_features):
    """Internal benchmark helper."""
    import torch
    from Two_level_Arch.src.feature_engineering import extract_heuristic_features

    N = data.shape[0]
    heur_feats = extract_heuristic_features(data, num_devices, exclude_last=True)

    # Heuristic
    t0 = time.perf_counter()
    h_scores = heuristic.score_batch(
        heur_feats["recency"], heur_feats["frequency"], heur_feats["power"]
    )
    heur_ms = (time.perf_counter() - t0) * 1000

    # XGBoost
    X_xgb = np.random.rand(N * num_devices, n_features).astype(np.float32)
    t0 = time.perf_counter()
    xgb_scores = xgb_model.predict(X_xgb).reshape(N, num_devices)
    xgb_ms = (time.perf_counter() - t0) * 1000

    # LSTM
    X_lstm = data[:, :-1, :].astype(np.float32)
    X_lstm[:, :, 0] /= 6.0
    X_lstm[:, :, 1] /= 7.0
    X_lstm[:, :, 2] /= max(X_lstm[:, :, 2].max(), 1)
    X_lstm[:, :, 3] /= max(X_lstm[:, :, 3].max(), 1)
    X_lstm[:, :, 4] /= max(X_lstm[:, :, 4].max(), 1)
    X_tensor = torch.FloatTensor(X_lstm)

    lstm_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        lstm_scores = torch.softmax(lstm_model(X_tensor), dim=-1).numpy()
    lstm_ms = (time.perf_counter() - t0) * 1000

    # Fusion
    t0 = time.perf_counter()
    fuser.predict(h_scores, xgb_scores, lstm_scores, top_k=5)
    fuse_ms = (time.perf_counter() - t0) * 1000

    total_ms = heur_ms + xgb_ms + lstm_ms + fuse_ms

    print(f"\n  Batch size: {N}")
    print(f"  {'Component':<25} {'Total (ms)':>12} {'Per-instance (ms)':>18}")
    print(f"  {'-'*25} {'-'*12} {'-'*18}")
    print(f"  {'Heuristic':<25} {heur_ms:>12.2f} {heur_ms/N:>18.4f}")
    print(f"  {'XGBoost':<25} {xgb_ms:>12.2f} {xgb_ms/N:>18.4f}")
    print(f"  {'LSTM':<25} {lstm_ms:>12.2f} {lstm_ms/N:>18.4f}")
    print(f"  {'Fusion':<25} {fuse_ms:>12.2f} {fuse_ms/N:>18.4f}")
    print(f"  {'TOTAL (end-to-end)':<25} {total_ms:>12.2f} {total_ms/N:>18.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE 2: KG-GNN (TensorFlow Lite)
# ═══════════════════════════════════════════════════════════════════════════

def run_kg_gnn_inference(benchmark: bool = False):
    """Run the KG-GNN inference pipeline.

    Attempts TFLite first; falls back to the full Keras model if TFLite
    requires Flex ops (tensorflow-select-tf-ops) that aren't installed.
    """
    print_header("Pipeline 2: KG-GNN Inference")

    tflite_path = os.path.join(PROJECT_ROOT, "KG_GNN", "models", "gnn_model.tflite")
    keras_path = os.path.join(PROJECT_ROOT, "KG_GNN", "models", "final_gnn_model.keras")

    try:
        import tensorflow as tf
    except ImportError:
        print("\n  ⚠ TensorFlow not installed. Install with:")
        print("    pip install tensorflow")
        return None

    # Prepare sample input
    sample = generate_sample_sequence()
    context = sample[:-1]
    target_device = int(sample[-1, 2])

    # Attempt TFLite first
    use_tflite = False
    if os.path.exists(tflite_path):
        print("\n  [1/3] Attempting TFLite model...")
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            use_tflite = True
            print(f"         ✓ TFLite model loaded")
        except RuntimeError as e:
            print(f"         ⚠ TFLite requires Flex ops (install tensorflow-select-tf-ops)")
            print(f"         Falling back to Keras model...")

    if use_tflite:
        # ── TFLite path ────────────────────────────────────────────────
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"         Inputs: {len(input_details)}")
        for i, inp in enumerate(input_details):
            print(f"           [{i}] {inp['name']}: shape={inp['shape']}, "
                  f"dtype={inp['dtype'].__name__}")

        print("\n  [2/3] Preparing sample input...")
        day_of_week = context[:, 0].astype(np.int32).reshape(1, -1)
        hour = context[:, 1].astype(np.int32).reshape(1, -1)
        device = context[:, 2].astype(np.int32).reshape(1, -1)
        control = context[:, 3].astype(np.float32).reshape(1, -1)
        device_control = context[:, 4].astype(np.int32).reshape(1, -1)
        inputs = [day_of_week, hour, device, control, device_control]

        print(f"         Ground truth: {DEVICE_TYPES.get(target_device)} (id={target_device})")

        print("\n  [3/3] Running TFLite inference...")
        t0 = time.perf_counter()
        for i, inp_data in enumerate(inputs):
            interpreter.set_tensor(input_details[i]['index'], inp_data)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
        inference_ms = (time.perf_counter() - t0) * 1000
        runtime_name = "TFLite"

    elif os.path.exists(keras_path):
        # ── Keras fallback ─────────────────────────────────────────────
        print(f"\n  [1/3] Loading Keras model: {os.path.basename(keras_path)}...")
        
        try:
            from KG_GNN.src.models.gnn_model import GCNEmbeddingLayer, ExpandFloatLayer
            custom_objs = {
                "GCNEmbeddingLayer": GCNEmbeddingLayer,
                "ExpandFloatLayer": ExpandFloatLayer
            }
        except ImportError:
            custom_objs = None
            print("         ⚠ Could not import custom GNN layers, loading might fail.")

        model = tf.keras.models.load_model(keras_path, custom_objects=custom_objs)
        print(f"         ✓ Keras model loaded ({model.count_params():,} parameters)")

        print("\n  [2/3] Preparing sample input...")
        print(f"         Ground truth: {DEVICE_TYPES.get(target_device)} (id={target_device})")

        # Build input dict matching model's expected input layers
        input_names = [layer.name for layer in model.input]
        print(f"         Model inputs: {input_names}")

        input_dict = {}
        for name in input_names:
            lower = name.lower()
            if "day" in lower:
                input_dict[name] = context[:, 0].astype(np.int32).reshape(1, -1)
            elif "hour" in lower:
                input_dict[name] = context[:, 1].astype(np.int32).reshape(1, -1)
            elif "device_control" in lower:
                input_dict[name] = context[:, 4].astype(np.int32).reshape(1, -1)
            elif "device" in lower:
                input_dict[name] = context[:, 2].astype(np.int32).reshape(1, -1)
            elif "control" in lower:
                input_dict[name] = context[:, 3].astype(np.float32).reshape(1, -1)
            else:
                input_dict[name] = context[:, 2].astype(np.int32).reshape(1, -1)

        print("\n  [3/3] Running Keras inference...")
        t0 = time.perf_counter()
        probs = model.predict(input_dict, verbose=0)[0]
        inference_ms = (time.perf_counter() - t0) * 1000
        runtime_name = "Keras"

    else:
        print(f"\n  ⚠ No model found. Expected one of:")
        print(f"    - {tflite_path}")
        print(f"    - {keras_path}")
        return None

    print(f"         Inference time: {inference_ms:.2f} ms ({runtime_name})")
    print(f"         Output probabilities shape: {probs.shape}")

    # Top-5 predictions
    top5_indices = np.argsort(probs)[::-1][:5]
    predictions = [
        (int(idx), DEVICE_TYPES.get(int(idx), f"device_{idx}"), float(probs[idx]))
        for idx in top5_indices
    ]
    print_predictions(predictions, title=f"★ KG-GNN Top-5 Predictions ({runtime_name})")

    hit = target_device in top5_indices
    print(f"\n  Ground truth: {DEVICE_TYPES.get(target_device)} (id={target_device})")
    print(f"  Hit@5: {'✅ YES' if hit else '❌ NO'}")

    # ── Save results ───────────────────────────────────────────────────
    result = {
        "pipeline": f"KG-GNN ({runtime_name})",
        "ground_truth": {"device_id": target_device,
                         "device_name": DEVICE_TYPES.get(target_device)},
        "predictions": [{"rank": i + 1, "device_id": int(d),
                         "device_name": DEVICE_TYPES.get(int(d)),
                         "score": float(probs[d])}
                        for i, d in enumerate(top5_indices)],
        "hit_at_5": bool(hit),
        "inference_ms": round(inference_ms, 4),
        "runtime": runtime_name,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "kg_gnn_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to output/kg_gnn_results.json")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SmartThings On-Device Predictive Model — Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_inference.py --all              Run both pipelines
  python demo_inference.py --two-level        Two-Level Architecture only
  python demo_inference.py --kg-gnn           KG-GNN only
  python demo_inference.py --all --benchmark  Both pipelines + latency benchmarks
        """
    )
    parser.add_argument("--two-level", action="store_true",
                        help="Run Two-Level Architecture (XGBoost + LSTM + Heuristic)")
    parser.add_argument("--kg-gnn", action="store_true",
                        help="Run KG-GNN (TensorFlow Lite)")
    parser.add_argument("--all", action="store_true",
                        help="Run both pipelines")
    parser.add_argument("--benchmark", action="store_true",
                        help="Include latency benchmarks")

    args = parser.parse_args()

    # Default to --all if no pipeline specified
    if not args.two_level and not args.kg_gnn and not args.all:
        args.all = True

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SmartThings On-Device Predictive Model — Inference Demo            ║")
    print("║  VIT_25ST07VIT Team                                                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    results = {}

    if args.two_level or args.all:
        results["two_level"] = run_two_level_inference(benchmark=args.benchmark)

    if args.kg_gnn or args.all:
        results["kg_gnn"] = run_kg_gnn_inference(benchmark=args.benchmark)

    # ── Summary ────────────────────────────────────────────────────────
    print_header("DEMO COMPLETE")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_path = os.path.join(OUTPUT_DIR, "demo_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  All results saved to: {summary_path}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
