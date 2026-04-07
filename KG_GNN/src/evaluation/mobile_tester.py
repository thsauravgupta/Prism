import tensorflow as tf
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity


# KERAS_MODEL_PATH = "../../models/best_gnn_model.keras"
# TFLITE_MODEL_PATH = "../../models/gnn_model.tflite"


interpreter = tf.lite.Interpreter(model_path="models/gnn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def extract_sample(X, i):
    return {
        "dayofweek": X[i:i+1, :, 0].astype(np.float32), 
        "hour": X[i:i+1, :, 1].astype(np.float32),      
        "device": X[i:i+1, :, 2].astype(np.float32),     
        "unknown": X[i:i+1, :, 3].astype(np.float32),
        "device_control": X[i:i+1, :, 4].astype(np.float32),  
    }


def compare_models(keras_model, tflite_path, sample):

    # Load TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set inputs
    for detail in input_details:
        name = detail['name']

        if "dayofweek" in name:
            interpreter.set_tensor(detail['index'], sample["dayofweek"])
        elif "hour" in name:
            interpreter.set_tensor(detail['index'], sample["hour"])
        elif "device" in name and "control" not in name:
            interpreter.set_tensor(detail['index'], sample["device"])
        elif "unknown" in name:
            interpreter.set_tensor(detail['index'], sample["unknown"])
        elif "device_control" in name:
            interpreter.set_tensor(detail['index'], sample["device_control"])

    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_details[0]['index'])

    keras_out = keras_model.predict(sample, verbose=0)

    diff = np.mean(np.abs(keras_out - tflite_out))

    print("Mean absolute difference:", diff)

    return diff


def evaluate_tflite(interpreter, X, y):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0

    for i in range(len(y)):
        sample = extract_sample(X, i)

        for detail in input_details:
            name = detail['name']
            dtype = detail['dtype']

            if "dayofweek" in name:
                key = "dayofweek"
            elif "hour" in name:
                key = "hour"
            elif "device_control" in name:
                key = "device_control"
            elif "device" in name:
                key = "device"
            elif "unknown" in name:
                key = "unknown"
            else:
                raise ValueError(f"Unknown input name: {name}")

            val = sample[key].astype(dtype)
            interpreter.set_tensor(detail['index'], val)

        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        pred = np.argmax(output)

        if pred == y[i]:
            correct += 1

    print("Accuracy:", correct / len(y))


def evaluate_topk_tflite(interpreter, X, y, k_values=[3, 5], max_samples=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    topk_correct = {k: 0 for k in k_values}
    total = len(y) if max_samples is None else min(len(y), max_samples)

    for i in range(total):
        sample = {
            "dayofweek": X[i:i+1, :, 0],
            "hour": X[i:i+1, :, 1],
            "device": X[i:i+1, :, 2],
            "unknown": X[i:i+1, :, 3],
            "device_control": X[i:i+1, :, 4],
        }

        # ---- set inputs ----
        for detail in input_details:
            name = detail['name']
            dtype = detail['dtype']

            if "dayofweek" in name:
                key = "dayofweek"
            elif "hour" in name:
                key = "hour"
            elif "device_control" in name:
                key = "device_control"
            elif "device" in name:
                key = "device"
            elif "unknown" in name:
                key = "unknown"
            else:
                raise ValueError(f"Unknown input: {name}")

            interpreter.set_tensor(detail['index'], sample[key].astype(dtype))

        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # ---- get top-k indices ----
        sorted_indices = np.argsort(output)[::-1]  # descending

        for k in k_values:
            if y[i] in sorted_indices[:k]:
                topk_correct[k] += 1

    # ---- print results ----
    for k in k_values:
        acc = topk_correct[k] / total
        print(f"Top-{k} Accuracy: {acc:.4f}")

    return {k: topk_correct[k] / total for k in k_values}

def benchmark(interpreter, sample, runs=100):
    input_details = interpreter.get_input_details()

    # set input once
    for detail in input_details:
        name = detail['name']

        if "dayofweek" in name:
            interpreter.set_tensor(detail['index'], sample["dayofweek"])
        elif "hour" in name:
            interpreter.set_tensor(detail['index'], sample["hour"])
        elif "device" in name and "control" not in name:
            interpreter.set_tensor(detail['index'], sample["device"])
        elif "unknown" in name:
            interpreter.set_tensor(detail['index'], sample["unknown"])
        elif "device_control" in name:
            interpreter.set_tensor(detail['index'], sample["device_control"])

    # warmup
    for _ in range(5):
        interpreter.invoke()

    times = []
    for _ in range(runs):
        start = time.time()
        interpreter.invoke()
        times.append(time.time() - start)

    print("Avg latency (ms):", np.mean(times) * 1000)
    
    
def cosine_similarity_models(keras_model, tflite_path, X, num_samples=100):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    similarities = []

    for i in range(num_samples):
        sample = {
            "dayofweek": X[i:i+1, :, 0],
            "hour": X[i:i+1, :, 1],
            "device": X[i:i+1, :, 2],
            "unknown": X[i:i+1, :, 3],
            "device_control": X[i:i+1, :, 4],
        }

        # ---- TFLite prediction ----
        for detail in input_details:
            name = detail['name']
            dtype = detail['dtype']

            if "dayofweek" in name:
                key = "dayofweek"
            elif "hour" in name:
                key = "hour"
            elif "device_control" in name:
                key = "device_control"
            elif "device" in name:
                key = "device"
            elif "unknown" in name:
                key = "unknown"
            else:
                raise ValueError(f"Unknown input: {name}")

            val = sample[key].astype(dtype)
            interpreter.set_tensor(detail['index'], val)

        interpreter.invoke()
        tflite_out = interpreter.get_tensor(output_details[0]['index'])

        # ---- Keras prediction ----
        keras_out = keras_model.predict(sample, verbose=0)

        # ---- Cosine similarity ----
        sim = cosine_similarity(keras_out, tflite_out)[0][0]
        similarities.append(sim)

    avg_sim = np.mean(similarities)

    print(f"Average Cosine Similarity: {avg_sim:.6f}")
    return avg_sim