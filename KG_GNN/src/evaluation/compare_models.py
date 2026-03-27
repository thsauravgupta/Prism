from typing import Dict, Callable, Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_ranking_metrics
from src.evaluation.latency import (
    estimate_keras_model_size_mb,
    benchmark_keras_inference,
)


def evaluate_model_bundle(
    models: Dict[str, Any],
    X_test_features: dict,
    y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test_features, verbose=0)
        rank_metrics = compute_ranking_metrics(y_test, y_pred)
        latency_metrics = benchmark_keras_inference(
            model=model,
            sample_features={k: v[:1] for k, v in X_test_features.items()},
            num_warmup=20,
            num_runs=100,
        )
        size_mb = estimate_keras_model_size_mb(model)

        row = {"Model": model_name, "ModelSizeMB": size_mb}
        row.update(rank_metrics)
        row.update(
            {
                "InferenceMeanMs": latency_metrics["mean_ms"],
                "InferenceP95Ms": latency_metrics["p95_ms"],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)