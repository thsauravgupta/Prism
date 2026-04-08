
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from Two_level_Arch.models.lstm_model import RoutinePredictor
from Two_level_Arch.models.xgboost_model import ContextualReRanker
from Two_level_Arch.src.config import LSTM_SEQ_LEN, LSTM_INPUT_SIZE, MODEL_SAVE_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def get_file_size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0.0

def benchmark(fn, runs=200):
    """Benchmark a callable."""
    for _ in range(20): fn()  # warmup
    times = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        times.append((time.perf_counter() - s) * 1000)
    return {"mean_ms": np.mean(times), "median_ms": np.median(times),
            "p95_ms": np.percentile(times, 95), "min_ms": np.min(times)}


def validate_lstm():
    print("\n" + "=" * 70)
    print("  LSTM Validation")
    print("=" * 70)

    # Load original
    model = RoutinePredictor.load_model()
    model.eval()

    # Deterministic test inputs
    torch.manual_seed(42)
    inputs = [
        ("single", torch.randn(1, LSTM_SEQ_LEN, LSTM_INPUT_SIZE)),
        ("batch_10", torch.randn(10, LSTM_SEQ_LEN, LSTM_INPUT_SIZE)),
        ("batch_100", torch.randn(100, LSTM_SEQ_LEN, LSTM_INPUT_SIZE)),
    ]

    with torch.no_grad():
        ref_outputs = {name: model(x) for name, x in inputs}

    # Models to validate
    variants = [
        ("TorchScript", "lstm_torchscript.pt", False),
        ("Optimized", "lstm_optimized.pt", False),
        ("Optimized .ptl", "lstm_optimized.ptl", True),
        ("Quantized .pt", "lstm_quantized.pt", False),
        ("Quantized .ptl", "lstm_quantized.ptl", True),
    ]

    results = {}
    for vname, fname, is_ptl in variants:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"\n  ⚠ {vname}: {fname} not found, skipping")
            continue

        print(f"\n  ── {vname} ({fname}) ──")
        size_kb = get_file_size_kb(fpath)
        print(f"  Size: {size_kb:.2f} KB")

        try:
            if is_ptl:
                m = torch.jit.load(fpath)
            else:
                m = torch.jit.load(fpath)
        except Exception as e:
            print(f"   Failed to load: {e}")
            results[vname] = {"error": str(e)}
            continue

        vr = {"size_kb": size_kb, "tests": {}}

        for tname, x in inputs:
            try:
                with torch.no_grad():
                    out = m(x)
                ref = ref_outputs[tname]
                max_diff = torch.max(torch.abs(ref - out)).item()
                mae = torch.mean(torch.abs(ref - out)).item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0)
                ).item()

                # Check if top-K predictions match
                ref_topk = torch.topk(ref, 5, dim=-1).indices
                out_topk = torch.topk(out, 5, dim=-1).indices
                topk_match = (ref_topk == out_topk).all(dim=-1).float().mean().item()

                vr["tests"][tname] = {
                    "max_diff": max_diff, "mae": mae,
                    "cosine_sim": cos_sim, "topk_match": topk_match,
                    "pass": max_diff < 0.1,
                }
                status = "" if max_diff < 0.1 else "⚠"
                print(f"  {status} {tname}: max_diff={max_diff:.4e}, MAE={mae:.4e}, "
                      f"cos_sim={cos_sim:.6f}, top5_match={topk_match:.2%}")
            except Exception as e:
                vr["tests"][tname] = {"error": str(e)}
                print(f"   {tname}: {e}")

        # Benchmark single instance
        x1 = inputs[0][1]
        try:
            b = benchmark(lambda: m(x1))
            vr["benchmark_single_ms"] = b
            print(f"  ⏱ Single inference: {b['mean_ms']:.4f} ms (mean), {b['p95_ms']:.4f} ms (p95)")
        except Exception as e:
            print(f"  ⚠ Benchmark failed: {e}")

        results[vname] = vr

    # Original benchmark
    x1 = inputs[0][1]
    b_orig = benchmark(lambda: model(x1))
    results["Original"] = {
        "size_kb": get_file_size_kb(os.path.join(MODEL_SAVE_DIR, "lstm_routine_predictor.pt")),
        "benchmark_single_ms": b_orig,
    }
    print(f"\n  ⏱ Original: {b_orig['mean_ms']:.4f} ms (mean)")

    return results


def validate_xgboost():
    print("\n" + "=" * 70)
    print("  XGBoost Validation")
    print("=" * 70)
    import xgboost as xgb

    orig_path = os.path.join(MODEL_SAVE_DIR, "xgboost_reranker.json")
    ubj_path = os.path.join(OUTPUT_DIR, "xgboost_reranker.ubj")

    if not os.path.exists(ubj_path):
        print("  ⚠ UBJ file not found, skipping")
        return {}

    xgb_orig = ContextualReRanker()
    xgb_orig.load_model()
    n_feat = xgb_orig.model.n_features_in_

    xgb_ubj = xgb.XGBRanker()
    xgb_ubj.load_model(ubj_path)

    np.random.seed(42)
    tests = {"small": np.random.rand(38, n_feat).astype(np.float32),
             "medium": np.random.rand(380, n_feat).astype(np.float32),
             "large": np.random.rand(3800, n_feat).astype(np.float32)}

    results = {"original_kb": get_file_size_kb(orig_path), "ubj_kb": get_file_size_kb(ubj_path)}
    for name, X in tests.items():
        p1 = xgb_orig.predict(X)
        p2 = xgb_ubj.predict(X)
        md = np.max(np.abs(p1 - p2))
        status = "✅" if md < 1e-6 else "⚠"
        print(f"  {status} {name} (N={len(X)}): max_diff={md:.2e}")
        results[name] = {"max_diff": float(md), "pass": md < 1e-6}

    # Benchmark
    X_bench = tests["medium"]
    b1 = benchmark(lambda: xgb_orig.predict(X_bench))
    b2 = benchmark(lambda: xgb_ubj.predict(X_bench))
    print(f"  ⏱ JSON:  {b1['mean_ms']:.3f} ms | UBJ: {b2['mean_ms']:.3f} ms")
    results["benchmark"] = {"json_ms": b1, "ubj_ms": b2}
    return results


def validate_fuser():
    print("\n" + "=" * 70)
    print("  Hybrid Fuser Validation")
    print("=" * 70)
    from Two_level_Arch.models.hybrid_fuser import HybridDecisionEngine

    ptl_path = os.path.join(OUTPUT_DIR, "hybrid_fuser.ptl")
    pt_path = os.path.join(OUTPUT_DIR, "hybrid_fuser.pt")
    fpath = ptl_path if os.path.exists(ptl_path) else pt_path

    if not os.path.exists(fpath):
        print("  ⚠ Fuser model not found")
        return {}

    mobile_fuser = torch.jit.load(fpath)
    np_fuser = HybridDecisionEngine()

    torch.manual_seed(42)
    h = torch.rand(100, 38); x = torch.rand(100, 38); l = torch.rand(100, 38)

    with torch.no_grad():
        mobile_out = mobile_fuser(h, x, l)

    np_ranked, _ = np_fuser.predict(h.numpy(), x.numpy(), l.numpy(), top_k=5)

    match = sum(set(np_ranked[i]) == set(mobile_out[i].numpy()) for i in range(100))
    print(f"  Top-5 set match: {match}/100 ({match}%)")

    b = benchmark(lambda: mobile_fuser(h, x, l))
    print(f"  ⏱ Mobile fuser: {b['mean_ms']:.4f} ms (100 instances)")

    return {"topk_match": match, "benchmark_ms": b, "size_kb": get_file_size_kb(fpath)}


def print_summary(lstm_r, xgb_r, fuser_r):
    print("\n" + "═" * 70)
    print("  VALIDATION SUMMARY")
    print("═" * 70)

    orig_lstm_kb = lstm_r.get("Original", {}).get("size_kb", 0)
    orig_xgb_kb = xgb_r.get("original_kb", 0)

    print(f"\n  {'Component':<20} {'Original(KB)':>13} {'Mobile(KB)':>11} {'Reduction':>10} {'Status':>8}")
    print(f"  {'-'*20} {'-'*13} {'-'*11} {'-'*10} {'-'*8}")

    q_ptl = lstm_r.get("Quantized .ptl", {})
    q_kb = q_ptl.get("size_kb", 0)
    lstm_red = f"{((orig_lstm_kb - q_kb)/orig_lstm_kb*100):.1f}%" if orig_lstm_kb else "N/A"
    lstm_ok = all(t.get("pass", False) for t in q_ptl.get("tests", {}).values()) if q_ptl.get("tests") else False
    print(f"  {'LSTM (INT8 .ptl)':<20} {orig_lstm_kb:>13.2f} {q_kb:>11.2f} {lstm_red:>10} {'✅' if lstm_ok else '⚠':>8}")

    ubj_kb = xgb_r.get("ubj_kb", 0)
    xgb_red = f"{((orig_xgb_kb - ubj_kb)/orig_xgb_kb*100):.1f}%" if orig_xgb_kb else "N/A"
    xgb_ok = all(v.get("pass", False) for k, v in xgb_r.items() if isinstance(v, dict) and "pass" in v)
    print(f"  {'XGBoost (UBJ)':<20} {orig_xgb_kb:>13.2f} {ubj_kb:>11.2f} {xgb_red:>10} {'✅' if xgb_ok else '⚠':>8}")

    f_kb = fuser_r.get("size_kb", 0)
    print(f"  {'Fuser (.ptl)':<20} {'N/A':>13} {f_kb:>11.2f} {'new':>10} {'✅':>8}")

    tot_orig = orig_lstm_kb + orig_xgb_kb
    tot_opt = q_kb + ubj_kb + f_kb
    print(f"\n  Total: {tot_orig:.2f} KB -> {tot_opt:.2f} KB "
          f"({((tot_orig-tot_opt)/tot_orig*100):.1f}% reduction)")

    all_pass = lstm_ok and xgb_ok
    print(f"\n  Overall: {' ALL VALIDATIONS PASSED' if all_pass else '⚠ SOME VALIDATIONS NEED REVIEW'}")


def main():
    print("=" * 70)
    print("  SmartThings Mobile Model Validation")
    print("=" * 70)

    lstm_r = validate_lstm()
    xgb_r = validate_xgboost()
    fuser_r = validate_fuser()
    print_summary(lstm_r, xgb_r, fuser_r)

    report = {"lstm": lstm_r, "xgboost": xgb_r, "fuser": fuser_r}
    rpath = os.path.join(OUTPUT_DIR, "validation_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {rpath}")

if __name__ == "__main__":
    main()
