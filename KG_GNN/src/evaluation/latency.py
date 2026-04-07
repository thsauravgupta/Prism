import os
import time
import tempfile
from typing import Dict, Any

import numpy as np
import tensorflow as tf


def estimate_keras_model_size_mb(model: tf.keras.Model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        temp_path = tmp.name
    try:
        model.save(temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return float(size_mb)


def benchmark_keras_inference(
    model: tf.keras.Model,
    sample_features: Dict[str, np.ndarray],
    num_warmup: int = 20,
    num_runs: int = 100,
) -> Dict[str, Any]:
    for _ in range(num_warmup):
        _ = model(sample_features, training=False)

    times_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(sample_features, training=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    times_ms = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
    }


def convert_to_tflite(model: tf.keras.Model, output_path: str, quantize_float16: bool = False) -> str:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path


def benchmark_tflite_inference(
    tflite_path: str,
    sample_features: Dict[str, np.ndarray],
    num_warmup: int = 20,
    num_runs: int = 100,
) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    feature_order = ["dayofweek", "hour", "device", "unknown", "device_control"]

    for detail, name in zip(input_details, feature_order):
        tensor = sample_features[name].astype(detail["dtype"])
        interpreter.set_tensor(detail["index"], tensor)
    for _ in range(num_warmup):
        interpreter.invoke()

    times_ms = []
    for _ in range(num_runs):
        for detail, name in zip(input_details, feature_order):
            tensor = sample_features[name].astype(detail["dtype"])
            interpreter.set_tensor(detail["index"], tensor)

        start = time.perf_counter()
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    times_ms = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "model_size_mb": float(os.path.getsize(tflite_path) / (1024 * 1024)),
    }