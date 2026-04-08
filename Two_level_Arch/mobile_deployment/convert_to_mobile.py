"""
convert_to_mobile.py — Steps 2, 3, 4: Convert, Optimize & Reduce Size

Pipeline:
  LSTM:     .pt -> eval -> TorchScript -> optimize_for_mobile -> INT8 quantization -> .ptl
  XGBoost:  .json -> compressed UBJ binary
  Fuser:    reimplemented as TorchScript module -> bundled .ptl
"""
import os, sys, json, shutil
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from Two_level_Arch.models.lstm_model import RoutinePredictor
from Two_level_Arch.models.xgboost_model import ContextualReRanker
from Two_level_Arch.src.config import (
    LSTM_SEQ_LEN, LSTM_INPUT_SIZE, MODEL_SAVE_DIR, FUSION_WEIGHTS, TOP_K,
)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_file_size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0.0


class MobileFuser(torch.nn.Module):
    """Pure-PyTorch reimplementation of HybridDecisionEngine for mobile."""
    def __init__(self, w_h=0.3, w_x=0.35, w_l=0.35, top_k=5):
        super().__init__()
        self.register_buffer("w_h", torch.tensor(w_h))
        self.register_buffer("w_x", torch.tensor(w_x))
        self.register_buffer("w_l", torch.tensor(w_l))
        self.top_k = top_k

    def _normalize(self, s: torch.Tensor) -> torch.Tensor:
        s_min = s.min(dim=-1, keepdim=True).values
        s_max = s.max(dim=-1, keepdim=True).values
        d = s_max - s_min
        d = torch.where(d == 0, torch.ones_like(d), d)
        return (s - s_min) / d

    def forward(self, h: torch.Tensor, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        fused = self.w_h * self._normalize(h) + self.w_x * self._normalize(x) + self.w_l * self._normalize(l)
        _, top_idx = torch.topk(fused, self.top_k, dim=-1)
        return top_idx


def convert_lstm():
    print("\n" + "=" * 70)
    print("  LSTM Conversion Pipeline")
    print("=" * 70)
    res = {}
    orig_path = os.path.join(MODEL_SAVE_DIR, "lstm_routine_predictor.pt")
    res["original_size_kb"] = get_file_size_kb(orig_path)

    # Load + eval
    model = RoutinePredictor.load_model()
    model.eval()
    ex = torch.randn(1, LSTM_SEQ_LEN, LSTM_INPUT_SIZE)
    with torch.no_grad():
        orig_out = model(ex)
    print(f"  Original output shape: {orig_out.shape}")

    # TorchScript trace
    print("  [2B] TorchScript tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    ts_path = os.path.join(OUTPUT_DIR, "lstm_torchscript.pt")
    traced.save(ts_path)
    res["torchscript_size_kb"] = get_file_size_kb(ts_path)
    print(f"  Saved: {ts_path} ({res['torchscript_size_kb']:.2f} KB)")

    # optimize_for_mobile
    print("  [3A] optimize_for_mobile...")
    from torch.utils.mobile_optimizer import optimize_for_mobile
    opt = optimize_for_mobile(traced)
    opt_path = os.path.join(OUTPUT_DIR, "lstm_optimized.pt")
    opt.save(opt_path)
    res["optimized_size_kb"] = get_file_size_kb(opt_path)

    # Optimized .ptl
    opt_ptl = os.path.join(OUTPUT_DIR, "lstm_optimized.ptl")
    opt._save_for_lite_interpreter(opt_ptl)
    res["optimized_ptl_size_kb"] = get_file_size_kb(opt_ptl)
    print(f"  Optimized .ptl: {res['optimized_ptl_size_kb']:.2f} KB")

    # Dynamic INT8 quantization
    print("  [4A] Dynamic INT8 quantization...")
    qmodel = torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        q_out = qmodel(ex)
    q_diff = torch.max(torch.abs(orig_out - q_out)).item()
    q_mae = torch.mean(torch.abs(orig_out - q_out)).item()
    print(f"  INT8 max diff: {q_diff:.4e}, MAE: {q_mae:.4e}")

    # Trace quantized
    with torch.no_grad():
        tq = torch.jit.trace(qmodel, ex)
    tq_path = os.path.join(OUTPUT_DIR, "lstm_quantized.pt")
    tq.save(tq_path)
    res["quantized_ts_size_kb"] = get_file_size_kb(tq_path)

    # Quantized .ptl
    q_ptl = os.path.join(OUTPUT_DIR, "lstm_quantized.ptl")
    try:
        oq = optimize_for_mobile(tq)
        oq._save_for_lite_interpreter(q_ptl)
    except Exception:
        tq._save_for_lite_interpreter(q_ptl)
    res["quantized_ptl_size_kb"] = get_file_size_kb(q_ptl)
    print(f"  Quantized .ptl: {res['quantized_ptl_size_kb']:.2f} KB")

    # FP16
    print("  [4E] FP16 variant...")
    m16 = RoutinePredictor.load_model(); m16.eval(); m16.half()
    ex16 = ex.half()
    with torch.no_grad():
        traced16 = torch.jit.trace(m16, ex16)
    fp16_path = os.path.join(OUTPUT_DIR, "lstm_fp16.pt")
    traced16.save(fp16_path)
    res["fp16_size_kb"] = get_file_size_kb(fp16_path)
    fp16_ptl = os.path.join(OUTPUT_DIR, "lstm_fp16.ptl")
    try:
        o16 = optimize_for_mobile(traced16)
        o16._save_for_lite_interpreter(fp16_ptl)
        res["fp16_ptl_size_kb"] = get_file_size_kb(fp16_ptl)
    except Exception:
        res["fp16_ptl_size_kb"] = res["fp16_size_kb"]
    print(f"  FP16 .ptl: {res.get('fp16_ptl_size_kb', res['fp16_size_kb']):.2f} KB")

    res["accuracy_diff"] = {
        "quantized_int8_max_diff": q_diff,
        "quantized_int8_mae": q_mae,
    }
    return res


def convert_xgboost():
    print("\n" + "=" * 70)
    print("  XGBoost Compression")
    print("=" * 70)
    res = {}
    orig = os.path.join(MODEL_SAVE_DIR, "xgboost_reranker.json")
    res["original_size_kb"] = get_file_size_kb(orig)

    xgb_model = ContextualReRanker()
    xgb_model.load_model()
    n_feat = xgb_model.model.n_features_in_
    X_test = np.random.rand(380, n_feat).astype(np.float32)
    orig_preds = xgb_model.predict(X_test)

    ubj = os.path.join(OUTPUT_DIR, "xgboost_reranker.ubj")
    xgb_model.model.save_model(ubj)
    res["ubj_size_kb"] = get_file_size_kb(ubj)

    import xgboost as xgb
    v = xgb.XGBRanker(); v.load_model(ubj)
    diff = np.max(np.abs(orig_preds - v.predict(X_test)))
    res["ubj_max_diff"] = float(diff)
    res["size_reduction_pct"] = round(100 - res["ubj_size_kb"]/res["original_size_kb"]*100, 2)

    print(f"  JSON: {res['original_size_kb']:.2f} KB -> UBJ: {res['ubj_size_kb']:.2f} KB ({res['size_reduction_pct']:.1f}% smaller)")
    shutil.copy2(orig, os.path.join(OUTPUT_DIR, "xgboost_reranker.json"))
    return res


def convert_fuser():
    print("\n" + "=" * 70)
    print("  Hybrid Fuser -> TorchScript")
    print("=" * 70)
    res = {}
    fuser = MobileFuser(FUSION_WEIGHTS["heuristic"], FUSION_WEIGHTS["xgboost"], FUSION_WEIGHTS["lstm"], TOP_K)
    fuser.eval()

    scripted = torch.jit.script(fuser)
    fp = os.path.join(OUTPUT_DIR, "hybrid_fuser.pt")
    scripted.save(fp)
    res["fuser_ts_size_kb"] = get_file_size_kb(fp)

    ptl = os.path.join(OUTPUT_DIR, "hybrid_fuser.ptl")
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        o = optimize_for_mobile(scripted)
        o._save_for_lite_interpreter(ptl)
        res["fuser_ptl_size_kb"] = get_file_size_kb(ptl)
    except Exception:
        res["fuser_ptl_size_kb"] = res["fuser_ts_size_kb"]
    print(f"  Fuser .ptl: {res.get('fuser_ptl_size_kb',0):.2f} KB")
    return res


def main():
    print("=" * 70)
    print("  SmartThings Mobile Conversion Pipeline")
    print("=" * 70)
    r = {"lstm": convert_lstm(), "xgboost": convert_xgboost(), "fuser": convert_fuser()}

    # Summary
    print("\n" + "=" * 70)
    print("  CONVERSION SUMMARY")
    print("=" * 70)
    lo = r["lstm"]["original_size_kb"]
    lq = r["lstm"]["quantized_ptl_size_kb"]
    xo = r["xgboost"]["original_size_kb"]
    xu = r["xgboost"]["ubj_size_kb"]
    fk = r["fuser"].get("fuser_ptl_size_kb", 0)

    variants = [
        ("Original .pt", lo), ("TorchScript", r["lstm"]["torchscript_size_kb"]),
        ("Optimized .ptl", r["lstm"]["optimized_ptl_size_kb"]),
        ("Quantized INT8 .ptl", lq), ("FP16 .ptl", r["lstm"].get("fp16_ptl_size_kb", 0)),
    ]
    print(f"\n  {'LSTM Variant':<25} {'Size(KB)':>10} {'Reduction':>10}")
    for n, s in variants:
        print(f"  {n:<25} {s:>10.2f} {((lo-s)/lo*100):>9.1f}%")

    tot_orig = lo + xo
    tot_opt = lq + xu + fk
    print(f"\n  Total original:  {tot_orig:.2f} KB")
    print(f"  Total optimized: {tot_opt:.2f} KB")
    print(f"  Overall reduction: {((tot_orig-tot_opt)/tot_orig*100):.1f}%")

    with open(os.path.join(OUTPUT_DIR, "conversion_report.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\n  Report saved to output/conversion_report.json")

if __name__ == "__main__":
    main()
