"""

Implements standard ranking-based evaluation metrics:
  - Hit@K: Whether the ground truth is in top-K predictions
  - MRR: Mean Reciprocal Rank
  - NDCG@K: Normalized Discounted Cumulative Gain
"""

import numpy as np
from typing import List, Dict, Optional


def hit_at_k(predicted_ranks: np.ndarray, actual: np.ndarray, k: int) -> float:
    
    N = len(actual)
    hits = 0

    for i in range(N):
        
        top_k_devices = np.argsort(predicted_ranks[i])[::-1][:k]
        if actual[i] in top_k_devices:
            hits += 1

    return hits / N


def mrr(predicted_ranks: np.ndarray, actual: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        predicted_ranks: shape [N, num_devices] — prediction scores.
        actual: shape [N] — ground truth device IDs.

    Returns:
        MRR score.
    """
    N = len(actual)
    rr_sum = 0.0

    for i in range(N):
        sorted_devices = np.argsort(predicted_ranks[i])[::-1]
        for rank, dev_id in enumerate(sorted_devices, start=1):
            if dev_id == actual[i]:
                rr_sum += 1.0 / rank
                break

    return rr_sum / N


def ndcg_at_k(predicted_ranks: np.ndarray, actual: np.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K.

    For single-target ranking (only one relevant item).

    Args:
        predicted_ranks: shape [N, num_devices] — prediction scores.
        actual: shape [N] — ground truth device IDs.
        k: top-K cutoff.

    Returns:
        NDCG@K score.
    """
    N = len(actual)
    ndcg_sum = 0.0

    for i in range(N):
        top_k = np.argsort(predicted_ranks[i])[::-1][:k]
        dcg = 0.0
        for rank, dev_id in enumerate(top_k, start=1):
            if dev_id == actual[i]:
                dcg += 1.0 / np.log2(rank + 1)
                break

        # Ideal DCG for single relevant item = 1 / log2(2) = 1.0
        idcg = 1.0
        ndcg_sum += dcg / idcg

    return ndcg_sum / N


def evaluate_all(
    predicted_ranks: np.ndarray,
    actual: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Run all evaluation metrics.

    Args:
        predicted_ranks: shape [N, num_devices] — prediction scores.
        actual: shape [N] — ground truth device IDs.
        k_values: list of K values for Hit@K and NDCG@K.

    Returns:
        Dictionary of metric_name → score.
    """
    results = {}

    # MRR
    results["MRR"] = mrr(predicted_ranks, actual)

    # Hit@K and NDCG@K
    for k in k_values:
        results[f"Hit@{k}"] = hit_at_k(predicted_ranks, actual, k)
        results[f"NDCG@{k}"] = ndcg_at_k(predicted_ranks, actual, k)

    return results


def print_results(results: Dict[str, float]) -> None:
    """Pretty print evaluation results."""
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in results.items():
        print(f"  {metric:<12s}: {value:.4f}")
    print("=" * 50)
