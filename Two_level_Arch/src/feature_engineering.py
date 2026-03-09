

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from collections import Counter


def compute_recency(instance: np.ndarray, num_devices: int) -> np.ndarray:
    """Compute recency score for each device in the instance.

    Recency = inverse of steps since last usage. More recent = higher score.

    Args:
        instance: shape [10, 5], one sequence of actions.
        num_devices: total number of unique devices.

    Returns:
        Array of shape [num_devices] with recency scores in [0, 1].
    """
    seq_len = instance.shape[0]
    recency = np.zeros(num_devices)

    for t in range(seq_len):
        device_id = int(instance[t, 2])  # device is index 2
        if device_id < num_devices:
            # More recent = higher score
            recency[device_id] = (t + 1) / seq_len  # normalized to [0, 1]

    return recency


def compute_frequency(instance: np.ndarray, num_devices: int) -> np.ndarray:
    """Compute frequency score for each device.

    Frequency = count of device appearances / total steps.

    Args:
        instance: shape [10, 5].
        num_devices: total number of unique devices.

    Returns:
        Array of shape [num_devices] with frequency in [0, 1].
    """
    seq_len = instance.shape[0]
    freq = np.zeros(num_devices)

    device_ids = instance[:, 2].astype(int)
    counts = Counter(device_ids)

    for dev_id, count in counts.items():
        if dev_id < num_devices:
            freq[dev_id] = count / seq_len

    return freq


def compute_power_proxy(instance: np.ndarray, num_devices: int) -> np.ndarray:
    """Compute a power/energy proxy for each device.

    Since SmartSense doesn't have explicit power data, we use
    device_control diversity as a proxy — devices with more diverse
    controls used are likely consuming more energy.

    Args:
        instance: shape [10, 5].
        num_devices: total number of unique devices.

    Returns:
        Array of shape [num_devices] with power proxy in [0, 1].
    """
    power = np.zeros(num_devices)
    device_controls: Dict[int, set] = {}

    for t in range(instance.shape[0]):
        dev_id = int(instance[t, 2])
        dc_id = int(instance[t, 4]) 
        if dev_id < num_devices:
            if dev_id not in device_controls:
                device_controls[dev_id] = set()
            device_controls[dev_id].add(dc_id)

    
    max_diversity = max((len(v) for v in device_controls.values()), default=1)
    for dev_id, controls in device_controls.items():
        power[dev_id] = len(controls) / max_diversity

    return power



def extract_heuristic_features(
    data: np.ndarray,
    num_devices: int,
) -> Dict[str, np.ndarray]:
    
    N = data.shape[0]
    recency = np.zeros((N, num_devices))
    frequency = np.zeros((N, num_devices))
    power = np.zeros((N, num_devices))

    for i in range(N):
        recency[i] = compute_recency(data[i], num_devices)
        frequency[i] = compute_frequency(data[i], num_devices)
        power[i] = compute_power_proxy(data[i], num_devices)

    return {
        "recency": recency,
        "frequency": frequency,
        "power": power,
    }


def extract_xgboost_features(
    data: np.ndarray,
    num_devices: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    N = data.shape[0]
    heuristic_feats = extract_heuristic_features(data, num_devices)

    features_list = []
    labels_list = []
    groups = []

    for i in range(N):
        instance = data[i]  # [10, 5]

        # Target is the last device in the sequence (for next-action prediction)
        target_device = int(instance[-1, 2])

        # Contextual features: hour and day distributions
        hours = instance[:, 1].astype(int)
        days = instance[:, 0].astype(int)
        avg_hour = np.mean(hours) / 23.0  # normalize

        day_dist = np.zeros(7)
        for d in days:
            if d < 7:
                day_dist[d] += 1
        day_dist = day_dist / max(day_dist.sum(), 1)

        for dev_id in range(num_devices):
            feat = np.concatenate([
                [heuristic_feats["recency"][i, dev_id]],
                [heuristic_feats["frequency"][i, dev_id]],
                [heuristic_feats["power"][i, dev_id]],
                [avg_hour],
                day_dist,
            ])
            features_list.append(feat)
            labels_list.append(1.0 if dev_id == target_device else 0.0)

        groups.append(num_devices)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)
    groups = np.array(groups, dtype=np.int32)

    return X, y, groups


def extract_lstm_sequences(
    data: np.ndarray,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare input/target pairs for LSTM routine predictor.

    Input: first 9 steps of each instance (shape [N, 9, 5])
    Target: device ID of the 10th step (shape [N])

    Args:
        data: shape [N, 10, 5].
        normalize: whether to normalize features.

    Returns:
        X: input sequences [N, 9, 5]
        y: target device IDs [N]
    """
    X = data[:, :-1, :].astype(np.float32)  # first 9 steps
    y = data[:, -1, 2].astype(np.int64)     # device ID of 10th step

    if normalize:
        # Normalize each feature column
        # day_of_week: /6, hour: /23, device/control/device_control
        X[:, :, 0] = X[:, :, 0] / 6.0   # day to [0,1]
        X[:, :, 1] = X[:, :, 1] / 23.0  # hour to [0,1]
        # device, control, device_contro
        max_device = max(X[:, :, 2].max(), 1)
        max_control = max(X[:, :, 3].max(), 1)
        max_dc = max(X[:, :, 4].max(), 1)
        X[:, :, 2] = X[:, :, 2] / max_device
        X[:, :, 3] = X[:, :, 3] / max_control
        X[:, :, 4] = X[:, :, 4] / max_dc

    return X, y


def get_num_devices(data: np.ndarray) -> int:
    """Infer the number of unique devices from a data tensor."""
    return int(data[:, :, 2].max()) + 1


def get_num_device_controls(data: np.ndarray) -> int:
    """Infer the number of unique device_controls from a data tensor."""
    return int(data[:, :, 4].max()) + 1
