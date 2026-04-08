

import numpy as np
from typing import List, Tuple, Optional, Dict


class HeuristicScorer:
   

    def __init__(self, w1: float = 0.4, w2: float = 0.4, w3: float = 0.2):
      
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def score(self, recency: float, frequency: float, power: float) -> float:
        
        return self.w1 * recency + self.w2 * frequency + self.w3 * power

    def score_batch(
        self,
        recency: np.ndarray,
        frequency: np.ndarray,
        power: np.ndarray,
    ) -> np.ndarray:
        
        return self.w1 * recency + self.w2 * frequency + self.w3 * power

    def rank_devices(
        self,
        recency: np.ndarray,
        frequency: np.ndarray,
        power: np.ndarray,
        device_names: Optional[Dict[int, str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[int, float, str]]:
        
        scores = self.score_batch(recency, frequency, power)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            name = device_names.get(idx, f"device_{idx}") if device_names else f"device_{idx}"
            results.append((int(idx), float(scores[idx]), name))

        return results

    def rank_batch(
        self,
        recency: np.ndarray,
        frequency: np.ndarray,
        power: np.ndarray,
    ) -> np.ndarray:
       
        return self.score_batch(recency, frequency, power)

    def get_weights(self) -> Dict[str, float]:
        """Return current weights as a dictionary."""
        return {"w1_recency": self.w1, "w2_frequency": self.w2, "w3_power": self.w3}

    def set_weights(self, w1: float, w2: float, w3: float) -> None:
        """Update the heuristic weights."""
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __repr__(self) -> str:
        return (
            f"HeuristicScorer(w1={self.w1:.2f}, w2={self.w2:.2f}, w3={self.w3:.2f})\n"
            f"  Score = {self.w1}·Recency + {self.w2}·Frequency + {self.w3}·Power"
        )
