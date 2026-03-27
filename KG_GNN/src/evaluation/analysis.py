import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def get_ranked_predictions(model, X, prepare_features_fn, top_k=10):
    features = prepare_features_fn(X)
    y_prob = model.predict(features, verbose=0)

    top_indices = np.argsort(-y_prob, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(y_prob, top_indices, axis=1)

    return top_indices, top_scores, y_prob


def show_sample_ranking(model, X, prepare_features_fn, sample_idx=0, top_k=10):
    top_indices, top_scores, _ = get_ranked_predictions(
        model,
        X[sample_idx:sample_idx + 1],
        prepare_features_fn,
        top_k=top_k
    )

    print(f"Top-{top_k} predictions for sample {sample_idx}:")
    for rank, (device_id, score) in enumerate(
        zip(top_indices[0], top_scores[0]), start=1
    ):
        print(f"{rank:2d}. device_{device_id}  score={score:.4f}")


# def topk_coverage(model, X_test, prepare_features_fn, top_k=5):
#     features = prepare_features_fn(X_test)
#     y_prob = model.predict(features, verbose=0)

#     topk = np.argsort(-y_prob, axis=1)[:, :top_k]

#     unique_predicted = np.unique(topk)
#     total_devices = y_prob.shape[1]

#     return {
#         "unique_predicted_devices": int(len(unique_predicted)),
#         "total_devices": int(total_devices),
#         "coverage_ratio": float(len(unique_predicted) / total_devices),
#     }


def topk_coverage(y_prob, top_k=5):
    topk = np.argsort(-y_prob, axis=1)[:, :top_k]

    unique_predicted = np.unique(topk)
    total_devices = y_prob.shape[1]

    return {
        "unique_predicted_devices": int(len(unique_predicted)),
        "total_devices": int(total_devices),
        "coverage_ratio": float(len(unique_predicted) / total_devices),
    }

def compute_device_priority_stats(model, X_test, y_test, prepare_features_fn, top_k=5):
    features = prepare_features_fn(X_test)
    y_prob = model.predict(features, verbose=0)

    top1 = np.argmax(y_prob, axis=1)
    topk = np.argsort(-y_prob, axis=1)[:, :top_k]

    num_devices = y_prob.shape[1]

    rows = []
    for dev_id in range(num_devices):
        gt_count = int(np.sum(y_test == dev_id))
        pred_top1_count = int(np.sum(top1 == dev_id))
        pred_topk_count = int(np.sum(np.any(topk == dev_id, axis=1)))
        avg_score = float(np.mean(y_prob[:, dev_id]))

        rows.append({
            "device_id": dev_id,
            "ground_truth_count": gt_count,
            "pred_top1_count": pred_top1_count,
            "pred_topk_count": pred_topk_count,
            "avg_pred_score": avg_score,
        })

    df = pd.DataFrame(rows)

    df["top1_bias_ratio"] = (df["pred_top1_count"] + 1) / (df["ground_truth_count"] + 1)
    df["topk_bias_ratio"] = (df["pred_topk_count"] + 1) / (df["ground_truth_count"] + 1)

    return df.sort_values("pred_top1_count", ascending=False)


def per_device_classification_report(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        zero_division=0
    )

    df = pd.DataFrame({
        "device_id": np.arange(len(precision)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    return df.sort_values("support", ascending=False)