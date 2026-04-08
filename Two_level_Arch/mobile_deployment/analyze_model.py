"""
analyze_model.py — Step 1: Comprehensive Model Analysis

Analyzes the LSTM RoutinePredictor and XGBoost ContextualReRanker models:
  - Architecture details and parameter counts
  - Input/output shapes and dependencies
  - Compute complexity (FLOPs estimation)
  - Model file sizes
  - Mobile operator compatibility check
  - Inference latency benchmarks
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from Two_level_Arch.models.lstm_model import RoutinePredictor
from Two_level_Arch.models.xgboost_model import ContextualReRanker
from Two_level_Arch.src.config import (
    LSTM_SEQ_LEN, LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS, LSTM_BIDIRECTIONAL, MODEL_SAVE_DIR,
)

# ── Output directory ────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_file_size_kb(path: str) -> float:
    """Get file size in KB."""
    if os.path.exists(path):
        return os.path.getsize(path) / 1024
    return 0.0


def count_parameters(model: nn.Module) -> dict:
    """Count total and per-layer parameters."""
    layer_info = []
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
        layer_info.append({
            "name": name,
            "shape": list(param.shape),
            "params": num,
            "dtype": str(param.dtype),
            "requires_grad": param.requires_grad,
            "size_kb": (num * param.element_size()) / 1024,
        })

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "total_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
        "layers": layer_info,
    }


def estimate_flops_lstm(input_size, hidden_size, num_layers, seq_len, batch_size=1, bidirectional=False):
    """Estimate FLOPs for LSTM forward pass.
    
    Each LSTM cell has 4 gates, each gate does:
      - input transform: hidden_size * input_size MACs
      - hidden transform: hidden_size * hidden_size MACs
    So per timestep per layer: 4 * hidden_size * (input_size + hidden_size) * 2 FLOPs
    """
    dirs = 2 if bidirectional else 1
    flops_per_timestep_layer0 = 4 * hidden_size * (input_size + hidden_size) * 2 * dirs
    flops_per_timestep_other = 4 * hidden_size * (hidden_size * dirs + hidden_size) * 2 * dirs

    total = seq_len * batch_size * flops_per_timestep_layer0
    total += seq_len * batch_size * flops_per_timestep_other * (num_layers - 1)
    return total


def estimate_flops_fc(layers_info):
    """Estimate FLOPs for fully connected layers (weight multiply + bias add)."""
    total = 0
    for layer in layers_info:
        if "fc" in layer["name"] and "weight" in layer["name"]:
            # shape is [out_features, in_features]
            out_f, in_f = layer["shape"]
            total += 2 * out_f * in_f  # multiply + add
    return total


def check_mobile_compatibility(model: nn.Module):
    """Check if model operations are supported by PyTorch Mobile."""
    issues = []
    supported_ops = {
        "LSTM", "Linear", "ReLU", "Dropout", "Sequential",
        "Softmax", "BatchNorm1d", "LayerNorm",
    }

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type == "RoutinePredictor":
            continue
        if module_type not in supported_ops and module_type != "Sequential":
            issues.append(f"  ⚠ Potentially unsupported: {name} ({module_type})")

    return issues


def benchmark_inference(model: nn.Module, input_tensor: torch.Tensor, runs: int = 100):
    """Benchmark inference latency."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "runs": runs,
    }


def analyze_xgboost():
    """Analyze the XGBoost model."""
    print("\n" + "=" * 70)
    print("  XGBoost ContextualReRanker Analysis")
    print("=" * 70)

    xgb_path = os.path.join(MODEL_SAVE_DIR, "xgboost_reranker.json")
    xgb_size = get_file_size_kb(xgb_path)

    xgb_model = ContextualReRanker()
    try:
        xgb_model.load_model()
        n_features = xgb_model.model.n_features_in_
        n_estimators = xgb_model.model.n_estimators
        max_depth = xgb_model.model.max_depth
        booster = xgb_model.model.get_booster()
        n_trees = len(booster.get_dump())

        info = {
            "model_type": "XGBRanker",
            "file_format": "JSON",
            "file_size_kb": round(xgb_size, 2),
            "n_features_in": n_features,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "actual_n_trees": n_trees,
            "objective": "rank:pairwise",
        }

        print(f"  File:           {xgb_path}")
        print(f"  File size:      {xgb_size:.2f} KB")
        print(f"  Features:       {n_features}")
        print(f"  N estimators:   {n_estimators}")
        print(f"  Max depth:      {max_depth}")
        print(f"  Actual trees:   {n_trees}")

        # Benchmark
        X_dummy = np.random.rand(1000 * 38, n_features).astype(np.float32)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = xgb_model.predict(X_dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        xgb_bench = {
            "batch_size": "1000×38",
            "mean_ms": round(np.mean(times), 3),
            "median_ms": round(np.median(times), 3),
            "min_ms": round(np.min(times), 3),
        }
        info["benchmark"] = xgb_bench
        print(f"  Inference:      {xgb_bench['mean_ms']:.3f} ms (1000×38 batch)")

        return info

    except Exception as e:
        print(f"  ❌ Error loading XGBoost model: {e}")
        return {"error": str(e), "file_size_kb": round(xgb_size, 2)}


def analyze_lstm():
    """Analyze the LSTM RoutinePredictor model."""
    print("\n" + "=" * 70)
    print("  LSTM RoutinePredictor Analysis")
    print("=" * 70)

    lstm_path = os.path.join(MODEL_SAVE_DIR, "lstm_routine_predictor.pt")
    lstm_size = get_file_size_kb(lstm_path)

    # Load model
    model = RoutinePredictor.load_model()
    model.eval()

    # Parameter analysis
    param_info = count_parameters(model)

    print(f"\n  ── Architecture ──")
    print(f"  Input size:     {model.input_size}")
    print(f"  Hidden size:    {model.hidden_size}")
    print(f"  Num layers:     {model.num_layers}")
    print(f"  Num devices:    {model.num_devices} (output classes)")
    print(f"  Bidirectional:  {model.bidirectional}")
    print(f"  Sequence len:   {LSTM_SEQ_LEN}")

    print(f"\n  ── Parameters ──")
    print(f"  Total params:      {param_info['total_params']:,}")
    print(f"  Trainable:         {param_info['trainable_params']:,}")
    print(f"  Model memory:      {param_info['total_size_mb']:.3f} MB")
    print(f"  File size (.pt):   {lstm_size:.2f} KB")

    print(f"\n  ── Per-Layer Breakdown ──")
    for layer in param_info["layers"]:
        print(f"    {layer['name']:40s} {str(layer['shape']):20s} "
              f"{layer['params']:>10,} params  ({layer['size_kb']:.2f} KB)")

    # FLOPs
    lstm_flops = estimate_flops_lstm(
        model.input_size, model.hidden_size, model.num_layers,
        LSTM_SEQ_LEN, batch_size=1, bidirectional=model.bidirectional
    )
    fc_flops = estimate_flops_fc(param_info["layers"])
    total_flops = lstm_flops + fc_flops

    print(f"\n  ── Compute Complexity ──")
    print(f"  LSTM FLOPs:      {lstm_flops:,}")
    print(f"  FC FLOPs:        {fc_flops:,}")
    print(f"  Total FLOPs:     {total_flops:,}")
    print(f"  Total MFLOPs:    {total_flops / 1e6:.2f}")

    # Mobile compatibility
    issues = check_mobile_compatibility(model)
    print(f"\n  ── Mobile Compatibility ──")
    if issues:
        for issue in issues:
            print(f"    {issue}")
    else:
        print("    ✅ All operations are mobile-compatible")

    # Check specific op support
    print(f"\n  ── Module Hierarchy ──")
    for name, module in model.named_modules():
        if name:
            print(f"    {name:40s} → {type(module).__name__}")

    # Benchmark
    print(f"\n  ── Inference Benchmark (single instance) ──")
    x_single = torch.randn(1, LSTM_SEQ_LEN, LSTM_INPUT_SIZE)
    bench_single = benchmark_inference(model, x_single, runs=200)
    print(f"    Mean:    {bench_single['mean_ms']:.4f} ms")
    print(f"    Median:  {bench_single['median_ms']:.4f} ms")
    print(f"    P95:     {bench_single['p95_ms']:.4f} ms")
    print(f"    P99:     {bench_single['p99_ms']:.4f} ms")

    print(f"\n  ── Inference Benchmark (batch=100) ──")
    x_batch = torch.randn(100, LSTM_SEQ_LEN, LSTM_INPUT_SIZE)
    bench_batch = benchmark_inference(model, x_batch, runs=100)
    print(f"    Mean:    {bench_batch['mean_ms']:.4f} ms")
    print(f"    Median:  {bench_batch['median_ms']:.4f} ms")
    print(f"    P95:     {bench_batch['p95_ms']:.4f} ms")

    return {
        "architecture": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "num_devices": model.num_devices,
            "bidirectional": model.bidirectional,
            "sequence_length": LSTM_SEQ_LEN,
        },
        "parameters": {
            "total": param_info["total_params"],
            "trainable": param_info["trainable_params"],
            "model_memory_mb": round(param_info["total_size_mb"], 4),
        },
        "file_size_kb": round(lstm_size, 2),
        "flops": {
            "lstm": lstm_flops,
            "fc": fc_flops,
            "total": total_flops,
            "total_mflops": round(total_flops / 1e6, 2),
        },
        "mobile_compatible": len(issues) == 0,
        "compatibility_issues": issues,
        "benchmark_single": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in bench_single.items()},
        "benchmark_batch100": {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in bench_batch.items()},
        "layers": param_info["layers"],
    }


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       SmartThings Model Analysis — Mobile Deployment Prep          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    results = {}

    # Analyze LSTM
    results["lstm"] = analyze_lstm()

    # Analyze XGBoost
    results["xgboost"] = analyze_xgboost()

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    lstm_kb = results["lstm"]["file_size_kb"]
    xgb_kb = results["xgboost"].get("file_size_kb", 0)
    total_kb = lstm_kb + xgb_kb
    print(f"  LSTM model size:    {lstm_kb:.2f} KB")
    print(f"  XGBoost model size: {xgb_kb:.2f} KB")
    print(f"  Total:              {total_kb:.2f} KB ({total_kb/1024:.2f} MB)")
    print(f"  LSTM FLOPs:         {results['lstm']['flops']['total_mflops']:.2f} MFLOPs")
    print(f"  LSTM params:        {results['lstm']['parameters']['total']:,}")
    print(f"  Mobile compatible:  {'✅ Yes' if results['lstm']['mobile_compatible'] else '❌ No'}")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "analysis_report.json")

    # Convert non-serializable types
    serializable = json.loads(json.dumps(results, default=str))
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  📄 Full report saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
