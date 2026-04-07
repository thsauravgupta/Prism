# SmartThings Mobile Deployment Summary

## Models Generated

All optimized models are located in:
```
Two_level_Arch/mobile_deployment/output/
```

### Recommended Mobile Files

| File | Size | Purpose |
|---|---|---|
| `lstm_quantized.ptl` | **451 KB** | LSTM model — INT8 quantized, Lite Interpreter (Android + iOS) |
| `xgboost_reranker.ubj` | **788 KB** | XGBoost model — compressed binary format |
| `hybrid_fuser.ptl` | **5 KB** | Score fusion module — TorchScript Lite Interpreter |
| **Total** | **1,244 KB** | |

### Alternative LSTM Variants (also generated)

| Variant | File | Size | Notes |
|---|---|---|---|
| FP16 | `lstm_fp16.ptl` | 444 KB | Slightly smaller, needs FP16 runtime support |
| Quantized TorchScript | `lstm_quantized.pt` | 254 KB | Smallest raw size, standard TorchScript (non-lite) |
| Optimized FP32 | `lstm_optimized.ptl` | 879 KB | Full precision, mobile-optimized |
| Unoptimized TorchScript | `lstm_torchscript.pt` | 886 KB | Baseline TorchScript |

---

## Size Comparison

### LSTM RoutinePredictor

| Stage | Size (KB) | Reduction |
|---|---|---|
| Original `.pt` | 873.37 | — |
| TorchScript `.pt` | 885.91 | -1.4% (overhead) |
| Mobile Optimized `.ptl` | 878.74 | -0.6% |
| **INT8 Quantized `.ptl`** | **451.47** | **48.3%** |
| FP16 `.ptl` | 443.75 | 49.2% |
| INT8 Quantized `.pt` (non-lite) | 254.41 | 70.9% |

### XGBoost ContextualReRanker

| Format | Size (KB) | Reduction |
|---|---|---|
| Original JSON | 1,229.42 | — |
| **Compressed UBJ** | **788.24** | **35.9%** |

### Overall

| Metric | Original | Optimized | Reduction |
|---|---|---|---|
| LSTM | 873 KB | 451 KB | **48.3%** |
| XGBoost | 1,229 KB | 788 KB | **35.9%** |
| Fuser | N/A | 5 KB | new |
| **Total** | **2,103 KB** | **1,244 KB** | **40.8%** |

---

## Optimizations Applied

### LSTM (PyTorch → Mobile)
1. **Eval mode** — disabled dropout layers (training-only)
2. **TorchScript tracing** (`torch.jit.trace`) — compiled computation graph
3. **`optimize_for_mobile()`** — fused ops, removed dead code
4. **Dynamic INT8 quantization** — quantized LSTM + Linear layers to `qint8`
5. **Lite Interpreter export** (`.ptl`) — minimal runtime footprint

### XGBoost
1. **Universal Binary JSON (UBJ)** export — 35.9% smaller than JSON format
2. Zero accuracy loss (bit-identical predictions)

### Hybrid Fuser
1. **Reimplemented in pure PyTorch** — no NumPy dependency
2. **TorchScript scripted** (`torch.jit.script`) — portable to mobile
3. Uses `torch.topk` instead of numpy argsort for efficiency

---

## Accuracy Validation Results

### LSTM INT8 Quantized vs Original

| Metric | Single (N=1) | Batch (N=10) | Batch (N=100) |
|---|---|---|---|
| Max absolute diff | 0.087 | 0.139 | 0.297 |
| Mean absolute error | 0.031 | 0.032 | 0.040 |
| Cosine similarity | 0.99999 | 0.99999 | 0.99997 |
| Top-5 exact match | 100% | 90% | 89% |

> The INT8 quantization introduces minor numerical differences but maintains
> **>99.99% cosine similarity** and **~90% top-5 ranking agreement**.
> For a ranking/recommendation system, this is well within acceptable bounds.

### XGBoost UBJ vs Original JSON
- **Max difference: 0.0** (bit-identical across all test sizes)

### Mobile Fuser vs NumPy Fuser
- **Top-5 set match: 100/100** (perfect agreement)

---

## Performance Benchmarks

### LSTM Single Instance Inference (CPU)

| Variant | Mean (ms) | P95 (ms) |
|---|---|---|
| Original PyTorch | 1.66 | 2.12 |
| TorchScript | 1.22 | 1.49 |
| INT8 Quantized .ptl | 1.98 | 2.48 |
| FP16 .ptl | — | — |

### XGBoost (380 instances)

| Format | Mean (ms) |
|---|---|
| JSON | 0.72 |
| UBJ | 0.62 |

---

## Mobile Integration Guide

### Android (PyTorch Mobile)

```kotlin
// build.gradle
implementation 'org.pytorch:pytorch_android_lite:2.1.0'

// Load LSTM
val lstmModule = LiteModuleLoader.load(assetFilePath("lstm_quantized.ptl"))
val input = Tensor.fromBlob(floatArray, longArrayOf(1, 9, 5))
val output = lstmModule.forward(IValue.from(input)).toTensor()

// Load Fuser
val fuserModule = LiteModuleLoader.load(assetFilePath("hybrid_fuser.ptl"))
```

### iOS (PyTorch Mobile)

```swift
// Podfile
pod 'LibTorch_Lite', '~> 2.1.0'

// Load LSTM
let lstmModule = try TorchModule(fileAtPath: lstmPath)
let input = Tensor(data: inputData, shape: [1, 9, 5])
let output = lstmModule.predict(with: input)

// Load Fuser
let fuserModule = try TorchModule(fileAtPath: fuserPath)
```

### XGBoost (Native)
XGBoost uses its native C API for mobile. Load the `.ubj` file:
- **Android**: Use JNI binding to XGBoost C library
- **iOS**: Link XGBoost C library via framework

---

## Architecture

```
Input: [batch, 9, 5] (9 time steps × 5 features)
        │
  ┌─────▼─────┐     ┌──────────────┐
  │ LSTM .ptl  │     │ XGBoost .ubj │     Heuristic
  │ (451 KB)   │     │ (788 KB)     │     (computed)
  └─────┬──────┘     └──────┬───────┘        │
        │                   │                │
    lstm_scores         xgb_scores      heur_scores
        │                   │                │
        └───────────┬───────┘                │
                    │                        │
              ┌─────▼────────────────────────▼───┐
              │        Fuser .ptl (5 KB)         │
              │  0.35×lstm + 0.35×xgb + 0.3×heur │
              └─────────────┬────────────────────┘
                            │
                    Top-5 Device IDs
```
