

import numpy as np
from typing import Dict, List, Tuple, Optional


class HybridDecisionEngine:
    

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        
        if weights is None:
            from Two_level_Arch.src.config import FUSION_WEIGHTS
            weights = FUSION_WEIGHTS.copy()

        self.weights = weights
        self._validate_weights()

    def _validate_weights(self) -> None:
        
        required = {"heuristic", "xgboost", "lstm"}
        missing = required - set(self.weights.keys())
        if missing:
            raise ValueError(f"Missing weight keys: {missing}")

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        
        s_min = scores.min(axis=-1, keepdims=True)
        s_max = scores.max(axis=-1, keepdims=True)
        denom = s_max - s_min
        denom = np.where(denom == 0, 1.0, denom)
        return (scores - s_min) / denom

    def fuse(
        self,
        heuristic_scores: np.ndarray,
        xgboost_scores: np.ndarray,
        lstm_scores: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        
        if normalize:
            heuristic_scores = self._normalize_scores(heuristic_scores.astype(np.float64))
            xgboost_scores = self._normalize_scores(xgboost_scores.astype(np.float64))
            lstm_scores = self._normalize_scores(lstm_scores.astype(np.float64))

        fused = (
            self.weights["heuristic"] * heuristic_scores
            + self.weights["xgboost"] * xgboost_scores
            + self.weights["lstm"] * lstm_scores
        )

        return fused

    def rank(
        self,
        fused_scores: np.ndarray,
        top_k: int = 5,
    ) -> np.ndarray:
        
        if fused_scores.ndim == 1:
            return np.argsort(fused_scores)[::-1][:top_k]

        # Batch mode
        N = fused_scores.shape[0]
        results = np.zeros((N, top_k), dtype=np.int64)
        for i in range(N):
            results[i] = np.argsort(fused_scores[i])[::-1][:top_k]

        return results

    def predict(
        self,
        heuristic_scores: np.ndarray,
        xgboost_scores: np.ndarray,
        lstm_scores: np.ndarray,
        top_k: int = 5,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        fused = self.fuse(heuristic_scores, xgboost_scores, lstm_scores, normalize)
        ranked = self.rank(fused, top_k)
        return ranked, fused

    def fuse_two(
        self,
        heuristic_scores: np.ndarray,
        model_scores: np.ndarray,
        heuristic_weight: float = 0.3,
        model_weight: float = 0.7,
        normalize: bool = True,
    ) -> np.ndarray:
        
        if normalize:
            heuristic_scores = self._normalize_scores(heuristic_scores.astype(np.float64))
            model_scores = self._normalize_scores(model_scores.astype(np.float64))

        return heuristic_weight * heuristic_scores + model_weight * model_scores

    def get_weights(self) -> Dict[str, float]:
        """Return current fusion weights."""
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update fusion weights."""
        self.weights = weights.copy()
        self._validate_weights()

    def __repr__(self) -> str:
        return (
            f"HybridDecisionEngine(\n"
            f"  heuristic_weight={self.weights['heuristic']:.2f},\n"
            f"  xgboost_weight={self.weights['xgboost']:.2f},\n"
            f"  lstm_weight={self.weights['lstm']:.2f}\n"
            f")"
        )
