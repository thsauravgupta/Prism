

import os
import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any


class ContextualReRanker:
  

    def __init__(self, params: Optional[Dict[str, Any]] = None):
       
        if params is None:
            from Two_level_Arch.src.config import XGB_PARAMS
            params = XGB_PARAMS.copy()

        self.params = params
        self.is_trained = False

        # Extract XGBRanker-compatible params
        self.model = xgb.XGBRanker(
            objective=params.get("objective", "rank:pairwise"),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 6),
            n_estimators=params.get("n_estimators", 200),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 5),
            gamma=params.get("gamma", 0.1),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
            verbosity=params.get("verbosity", 1),
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        groups_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the XGBoost re-ranker.

        Args:
            X_train: Training feature matrix [N_train, num_features].
            y_train: Training labels [N_train].
            groups_train: Group sizes for ranking [num_groups_train].
            X_val: Optional validation features.
            y_val: Optional validation labels.
            groups_val: Optional validation group sizes.
        """
        print(f"[INFO] Training XGBoost Re-Ranker...")
        print(f"  X_train: {X_train.shape}, groups: {groups_train.shape}")

        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "group": groups_train,
            "verbose": True,
        }

        # Add validation set if provided
        if X_val is not None and y_val is not None and groups_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_group"] = [groups_val]
            print(f"  X_val:   {X_val.shape}, groups: {groups_val.shape}")

        self.model.fit(**fit_kwargs)
        self.is_trained = True
        print("[INFO] XGBoost training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance scores for device candidates.

        Args:
            X: Feature matrix [N, num_features].

        Returns:
            Predicted scores [N].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_for_instance(
        self,
        X_instance: np.ndarray,
        num_devices: int,
    ) -> np.ndarray:
        """Predict scores for all devices for one instance.

        Args:
            X_instance: Feature matrix [num_devices, num_features].
            num_devices: Number of candidate devices.

        Returns:
            Score array [num_devices].
        """
        return self.predict(X_instance)

    def predict_batch(
        self,
        X: np.ndarray,
        num_devices: int,
    ) -> np.ndarray:
        """Predict for a batch of instances.

        Args:
            X: Feature matrix [N * num_devices, num_features].
            num_devices: Number of devices per instance.

        Returns:
            Score matrix [N, num_devices].
        """
        raw_scores = self.predict(X)
        N = len(raw_scores) // num_devices
        return raw_scores.reshape(N, num_devices)

    def save_model(self, path: Optional[str] = None) -> str:
        """Save model to file.

        Args:
            path: File path. If None, uses default from config.

        Returns:
            Path where model was saved.
        """
        if path is None:
            from Two_level_Arch.src.config import MODEL_SAVE_DIR
            path = os.path.join(MODEL_SAVE_DIR, "xgboost_reranker.json")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"[INFO] XGBoost model saved to {path}")
        return path

    def load_model(self, path: Optional[str] = None) -> None:
        """Load model from file.

        Args:
            path: File path. If None, uses default from config.
        """
        if path is None:
            from Two_level_Arch.src.config import MODEL_SAVE_DIR
            path = os.path.join(MODEL_SAVE_DIR, "xgboost_reranker.json")

        self.model.load_model(path)
        self.is_trained = True
        print(f"[INFO] XGBoost model loaded from {path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if model is trained."""
        if not self.is_trained:
            return None
        return dict(zip(
            [f"f{i}" for i in range(len(self.model.feature_importances_))],
            self.model.feature_importances_
        ))

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"ContextualReRanker(status={status}, params={self.params})"
