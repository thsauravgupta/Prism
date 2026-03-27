import numpy as np


def _safe_rank_of_true_label(scores: np.ndarray, true_label: int) -> int:
    """
    Returns 1-based rank of the true label in descending predicted scores.
    """
    order = np.argsort(-scores)
    rank = np.where(order == true_label)[0]
    if len(rank) == 0:
        return len(scores) + 1
    return int(rank[0]) + 1


def mean_reciprocal_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rr = []
    for t, s in zip(y_true, y_pred):
        rank = _safe_rank_of_true_label(s, int(t))
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def mean_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ranks = []
    for t, s in zip(y_true, y_pred):
        rank = _safe_rank_of_true_label(s, int(t))
        ranks.append(rank)
    return float(np.mean(ranks))


def median_rank(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ranks = []
    for t, s in zip(y_true, y_pred):
        rank = _safe_rank_of_true_label(s, int(t))
        ranks.append(rank)
    return float(np.median(ranks))


def coverage_ratio(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    covered = []
    for t, s in zip(y_true, y_pred):
        top_k = np.argsort(-s)[:k]
        covered.append(1.0 if int(t) in top_k else 0.0)
    return float(np.mean(covered))

def hit_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    hits = []
    for t, s in zip(y_true, y_pred):
        top_k = np.argsort(-s)[:k]
        hits.append(1.0 if int(t) in top_k else 0.0)
    return float(np.mean(hits))


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    vals = []
    for t, s in zip(y_true, y_pred):
        top_k = np.argsort(-s)[:k]
        if int(t) in top_k:
            rank = np.where(top_k == int(t))[0][0] + 1
            vals.append(1.0 / np.log2(rank + 1))
        else:
            vals.append(0.0)
    return float(np.mean(vals))


def compute_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)

    return {
        "MRR": mean_reciprocal_rank(y_true, y_pred),

        "Hit@1": hit_at_k(y_true, y_pred, 1),
        "NDCG@1": ndcg_at_k(y_true, y_pred, 1),

        "Hit@3": hit_at_k(y_true, y_pred, 3),
        "NDCG@3": ndcg_at_k(y_true, y_pred, 3),

        "Hit@5": hit_at_k(y_true, y_pred, 5),
        "NDCG@5": ndcg_at_k(y_true, y_pred, 5),

        "Hit@10": hit_at_k(y_true, y_pred, 10),
        "NDCG@10": ndcg_at_k(y_true, y_pred, 10),

        "mean_rank": mean_rank(y_true, y_pred),
        "median_rank": median_rank(y_true, y_pred),
        "coverage_ratio": coverage_ratio(y_true, y_pred, k=10),
    }