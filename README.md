# SmartThings On-Device Predictive Model Integration

## Project Overview

This project implements **on-device predictive models** for the Samsung SmartThings ecosystem. Given a user's recent sequence of smart-home device interactions, the system predicts which device the user is most likely to use next.

Two independent predictive pipelines are implemented:

| Pipeline | Technique | Mobile Runtime | Model Size |
|---|---|---|---|
| **Two-Level Architecture** | Heuristic + XGBoost + LSTM → Weighted Fusion | PyTorch Mobile (`.ptl`) | ~1.2 MB |
| **KG-GNN** | Knowledge Graph + Graph Neural Network | TensorFlow Lite (`.tflite`) | ~95 KB |

---

## Project Structure

```
├── Two_level_Arch/                  # Pipeline 1: Two-Level Architecture
│   ├── models/                      # Model definitions
│   │   ├── heuristic.py             #   Rule-based scorer (recency/frequency/power)
│   │   ├── xgboost_model.py         #   XGBoost learning-to-rank re-ranker
│   │   ├── lstm_model.py            #   LSTM sequence predictor
│   │   └── hybrid_fuser.py          #   Weighted score fusion engine
│   ├── src/                         # Data pipeline & utilities
│   │   ├── config.py                #   Central configuration
│   │   ├── data_loader.py           #   SmartSense data loading
│   │   ├── dataset.py               #   PyTorch Dataset class
│   │   ├── feature_engineering.py   #   Feature extraction
│   │   └── evaluation.py            #   Hit@K, MRR, NDCG metrics
│   ├── saved_models/                # Pre-trained model weights
│   │   ├── lstm_routine_predictor.pt
│   │   └── xgboost_reranker.json
│   ├── mobile_deployment/           # Mobile conversion pipeline
│   │   ├── analyze_model.py         #   Model profiling
│   │   ├── convert_to_mobile.py     #   PyTorch → TorchScript → .ptl
│   │   ├── validate_mobile_model.py #   Accuracy validation
│   │   └── deployment_summary.md    #   Conversion results
│   └── training_notebook.ipynb      # Training notebook
│
├── KG_GNN/                          # Pipeline 2: Knowledge Graph GNN
│   ├── src/                         #   GNN source code
│   ├── models/                      #   Trained models
│   │   ├── gnn_model.tflite         #   TFLite model (95 KB)
│   │   ├── best_gnn_model.keras     #   Keras checkpoint
│   │   └── final_gnn_model.keras    #   Final Keras model
│   ├── main.py                      #   Training + TFLite export
│   └── KG_GNN_model.ipynb           #   Training notebook
│
├── mobile_app/                      # Android app (Kotlin/Compose)
│   ├── app/src/main/
│   │   ├── assets/                  #   Model files (.ptl, .tflite)
│   │   └── java/.../inference/      #   On-device inference engines
│   └── build.gradle.kts
│
├── demo_inference.py                # ★ Standalone inference demo script
├── Dockerfile                       # Docker container definition
├── docker-compose.yml               # Docker Compose orchestration
├── requirements.txt                 # Full Python dependencies
├── requirements-inference.txt       # Inference-only dependencies
└── README.md                        # This file
```

---

## Quick Start

### Option 1: Docker (Recommended — Zero Setup)

```bash
# Clone the repository
git clone <repo-url>
cd VIT_25ST07VIT_On-Device_Predictive_Model_Integration_for_SmartThings_Ecosystem

# Build and run the inference demo
docker compose up --build
```

This will:
- Build a Docker container with all dependencies
- Run `demo_inference.py --all` which executes both pipelines
- Save results to `./output/`

**Run specific pipelines:**
```bash
# Two-Level Architecture only
docker compose run inference python demo_inference.py --two-level

# KG-GNN only
docker compose run inference python demo_inference.py --kg-gnn

# Both pipelines with latency benchmarks
docker compose run inference python demo_inference.py --all --benchmark
```

### Option 2: Local Python Environment

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements-inference.txt

# 3. Run the demo
python demo_inference.py --all
```

### Option 3: Android App

```bash
cd mobile_app

# Create local.properties (adjust SDK path if different)
echo "sdk.dir=C\:\\Users\\<USERNAME>\\AppData\\Local\\Android\\Sdk" > local.properties

# Build the APK
./gradlew assembleDebug

# Install on connected device/emulator
./gradlew installDebug

# Launch the app
adb shell am start -n com.example.smartthingsdevices/.MainActivity
```

Or open `mobile_app/` in Android Studio and click **Run**.

---

## Inference Demo

The `demo_inference.py` script is the primary entry point for demonstrating both models. It requires **no modifications** to run.

### What It Does

1. **Generates a sample input** — a realistic 9-step smart-home routine (morning sequence with MotionSensor, Light, Television, Blind, etc.)
2. **Runs Pipeline 1 (Two-Level Architecture)**:
   - Computes heuristic scores (recency + frequency + power)
   - Runs XGBoost re-ranking
   - Runs LSTM sequence prediction
   - Fuses all scores with weighted combination (30% heuristic + 35% XGBoost + 35% LSTM)
   - Outputs Top-5 predicted devices
3. **Runs Pipeline 2 (KG-GNN)**:
   - Loads the TFLite model
   - Decomposes input into 5 separate feature tensors
   - Runs GNN inference
   - Outputs Top-5 predicted devices
4. **Saves results** to `output/` as JSON files

### Usage

```bash
python demo_inference.py --help

# Run everything
python demo_inference.py --all

# Run with latency benchmarks (1000-instance batch)
python demo_inference.py --all --benchmark

# Run specific pipeline
python demo_inference.py --two-level
python demo_inference.py --kg-gnn
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════════════╗
║  SmartThings On-Device Predictive Model — Inference Demo            ║
╚══════════════════════════════════════════════════════════════════════╝

======================================================================
  Pipeline 1: Two-Level Architecture Inference
======================================================================
  [1/5] Loading pre-trained models...
         ✓ HeuristicScorer loaded
         ✓ XGBoost ContextualReRanker loaded
         ✓ LSTM RoutinePredictor loaded
         ✓ HybridDecisionEngine loaded
  ...
  ★ Fused Top-5 Predictions
  Rank   Device                        Score
  #1     Television                    0.8234
  #2     Light                         0.6891
  ...
  Hit@5:  YES — target is in top-5

======================================================================
  Pipeline 2: KG-GNN Inference (TFLite)
======================================================================
  ...
  ★ KG-GNN Top-5 Predictions
  ...

======================================================================
  DEMO COMPLETE
======================================================================
  All results saved to: output/demo_results.json
```

---

## Docker

### Container Details

| Property | Value |
|---|---|
| Base Image | `python:3.11-slim` |
| Working Dir | `/app` |
| Exposed Volume | `/app/output` → `./output/` |
| Default CMD | `python demo_inference.py --all` |

### Build Commands

```bash
# Build the image
docker build -t smartthings-inference .

# Run interactively
docker run -it -v ./output:/app/output smartthings-inference

# Run with specific flags
docker run -v ./output:/app/output smartthings-inference \
    python demo_inference.py --all --benchmark

# Use docker compose (recommended)
docker compose up --build
```

### Rebuild After Changes

```bash
docker compose build --no-cache
docker compose up
```

---

## Model Details

### Pipeline 1: Two-Level Architecture

```
Input: [batch, 9, 5] (9 time steps × 5 features)
        │
  ┌─────▼─────┐     ┌──────────────┐
  │   LSTM     │     │   XGBoost    │     Heuristic
  │ (873 KB)   │     │ (1,229 KB)   │     (computed)
  └─────┬──────┘     └──────┬───────┘        │
        │                   │                │
    lstm_scores         xgb_scores      heur_scores
        │                   │                │
        └───────────┬───────┘                │
                    │                        │
              ┌─────▼────────────────────────▼───┐
              │     Hybrid Decision Engine        │
              │  0.35×lstm + 0.35×xgb + 0.3×heur │
              └─────────────┬────────────────────┘
                            │
                    Top-5 Device IDs
```

**Input format**: Each action = `[day_of_week, hour_bin, device_id, control_id, device_control_id]`

**Models**:
- **HeuristicScorer**: `Score = 0.4 × recency + 0.4 × frequency + 0.2 × power_proxy`
- **XGBoost ContextualReRanker**: `XGBRanker` with `rank:pairwise`, 200 trees, max_depth=6
- **LSTM RoutinePredictor**: 2-layer LSTM, hidden_size=128, dropout=0.3
- **HybridDecisionEngine**: Min-max normalisation → weighted sum → top-K

### Pipeline 2: KG-GNN

- **Architecture**: Graph Neural Network using device co-occurrence knowledge graph
- **Framework**: TensorFlow/Keras → exported to TFLite
- **Input**: 5 separate tensors (day, hour, device, control, device_control) each `[1, 9]`
- **Output**: Softmax probabilities over all device types

### Evaluation Metrics

Both pipelines are evaluated on:
- **Hit@K** (K=1,3,5,10): Is the ground truth in the top-K predictions?
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain

---

## Mobile App

The Android app (`mobile_app/`) integrates both models for on-device inference:

| Model | Runtime | Asset Files |
|---|---|---|
| Two-Level Arch | PyTorch Mobile Lite | `lstm_quantized.ptl`, `hybrid_fuser.ptl` |
| KG-GNN | TensorFlow Lite | `gnn_model.tflite` |

**Requirements**: Android Studio, API 24+, Kotlin

### Pre-built APK (Fastest for review)
For immediate evaluation without compiling, you can install the pre-built APK provided in the repository root:
```bash
# Install the pre-built APK to your connected device/emulator
adb install SmartThings_OnDevice_Predictive_Demo.apk

# Launch the app
adb shell am start -n com.example.smartthingsdevices/.MainActivity
```

### Build & Run from Source

```bash
cd mobile_app
./gradlew assembleDebug     # Build
./gradlew installDebug      # Install on device
```

Or open `mobile_app/` in Android Studio → select device → Run ▶

---

## Data

The models are trained on the [SmartSense dataset](https://github.com/snudatalab/SmartSense) which contains real Samsung SmartThings usage logs from 4 countries (Korea, USA, Spain, France).

Data is automatically downloaded by `Two_level_Arch/src/data_loader.py`:
```python
from Two_level_Arch.src.data_loader import download_smartsense
download_smartsense()  # Downloads and extracts to Two_level_Arch/data/
```

---

## Team

**VIT_25ST07VIT** — On-Device Predictive Model Integration for SmartThings Ecosystem
