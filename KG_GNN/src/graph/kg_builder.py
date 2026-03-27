import os
import pickle
import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KGBuilder:
    """
    Build a semantic device-device adjacency matrix from SmartThings logs.

    The resulting adjacency is device x device, so it remains compatible with
    the current GCNEmbeddingLayer / lightweight graph embedding approach.

    Signal sources:
    - shared day patterns
    - shared hour patterns
    - shared control patterns
    - shared device_control patterns
    - optional routine transition counts
    """

    def __init__(
        self,
        dictionary_manager,
        data_dir: str = "data",
        regions: List[str] = None,
        use_routines: bool = True,
        alpha_day: float = 0.15,
        alpha_hour: float = 0.25,
        alpha_control: float = 0.20,
        alpha_devctrl: float = 0.25,
        alpha_transition: float = 0.15,
        top_k: int = 8,
        add_self_loops: bool = True,
    ):
        self.dictionary_manager = dictionary_manager
        self.data_dir = data_dir
        self.regions = regions or ["fr", "kr", "sp", "us"]
        self.use_routines = use_routines

        self.alpha_day = alpha_day
        self.alpha_hour = alpha_hour
        self.alpha_control = alpha_control
        self.alpha_devctrl = alpha_devctrl
        self.alpha_transition = alpha_transition

        self.top_k = top_k
        self.add_self_loops = add_self_loops

    def build_adjacency_matrix(self) -> np.ndarray:
        global_dicts = self.dictionary_manager.global_dicts
        if not global_dicts["device_dict"]:
            self.dictionary_manager.build_global_dictionaries()
            global_dicts = self.dictionary_manager.global_dicts

        num_devices = len(global_dicts["device_dict"]) + 1
        num_days = len(global_dicts.get("dayofweek_dict", {})) + 1
        num_hours = len(global_dicts.get("hour_dict", {})) + 1
        num_controls = len(global_dicts.get("control_dict", {})) + 1
        num_devctrl = len(global_dicts.get("device_control_dict", {})) + 1

        # Device-context count matrices
        M_day = np.zeros((num_devices, num_days), dtype=np.float32)
        M_hour = np.zeros((num_devices, num_hours), dtype=np.float32)
        M_control = np.zeros((num_devices, num_controls), dtype=np.float32)
        M_devctrl = np.zeros((num_devices, num_devctrl), dtype=np.float32)

        # Optional transition matrix from routine corpus
        A_transition = np.zeros((num_devices, num_devices), dtype=np.float32)

        for region in self.regions:
            self._accumulate_log_statistics(
                region=region,
                M_day=M_day,
                M_hour=M_hour,
                M_control=M_control,
                M_devctrl=M_devctrl,
            )

            if self.use_routines:
                self._accumulate_routine_transitions(
                    region=region,
                    A_transition=A_transition,
                )

        # Device-device similarity from shared context
        A_day = self._normalize_similarity(M_day @ M_day.T)
        A_hour = self._normalize_similarity(M_hour @ M_hour.T)
        A_control = self._normalize_similarity(M_control @ M_control.T)
        A_devctrl = self._normalize_similarity(M_devctrl @ M_devctrl.T)

        if self.use_routines:
            A_transition = self._normalize_rows(A_transition + A_transition.T)
        else:
            A_transition = np.zeros_like(A_day)

        A = (
            self.alpha_day * A_day
            + self.alpha_hour * A_hour
            + self.alpha_control * A_control
            + self.alpha_devctrl * A_devctrl
            + self.alpha_transition * A_transition
        )

        # Remove node 0 from influencing others too much, but keep matrix shape
        A[0, :] = 0.0
        A[:, 0] = 0.0

        # Keep only top-k neighbors per node for sparsity and mobile-friendliness
        A = self._top_k_sparsify(A, k=self.top_k)

        if self.add_self_loops:
            np.fill_diagonal(A, A.diagonal() + 1.0)

        A = self._normalize_rows(A)

        logger.info("Generated semantic device adjacency matrix with shape: %s", A.shape)
        return A.astype(np.float32)

    def _accumulate_log_statistics(
        self,
        region: str,
        M_day: np.ndarray,
        M_hour: np.ndarray,
        M_control: np.ndarray,
        M_devctrl: np.ndarray,
    ) -> None:
        region_dir = os.path.join(self.data_dir, region)
        file_names = ["trn_instance_10.pkl", "vld_instance_10.pkl", "test_instance_10.pkl"]

        mappings = self.dictionary_manager.get_mappings(region)
        map_day = mappings.get("dayofweek_dict", {})
        map_hour = mappings.get("hour_dict", {})
        map_dev = mappings.get("device_dict", {})
        map_control = mappings.get("control_dict", {})
        map_devctrl = mappings.get("device_control_dict", {})

        for file_name in file_names:
            path = os.path.join(region_dir, file_name)
            if not os.path.exists(path):
                continue

            with open(path, "rb") as f:
                data = pickle.load(f)

            # Expected shape: [N x 10 x 5]
            # Action fields: day, hour, device, control, device_control
            for instance in data:
                for action in instance:
                    try:
                        day_id_reg = int(action[0])
                        hour_id_reg = int(action[1])
                        dev_id_reg = int(action[2])
                        control_id_reg = int(action[3])
                        devctrl_id_reg = int(action[4])
                    except (TypeError, ValueError, IndexError):
                        continue

                    dev_id = map_dev.get(dev_id_reg)
                    if dev_id is None or dev_id <= 0:
                        continue

                    day_id = map_day.get(day_id_reg)
                    hour_id = map_hour.get(hour_id_reg)
                    control_id = map_control.get(control_id_reg)
                    devctrl_id = map_devctrl.get(devctrl_id_reg)

                    if day_id is not None and day_id > 0:
                        M_day[dev_id, day_id] += 1.0
                    if hour_id is not None and hour_id > 0:
                        M_hour[dev_id, hour_id] += 1.0
                    if control_id is not None and control_id > 0:
                        M_control[dev_id, control_id] += 1.0
                    if devctrl_id is not None and devctrl_id > 0:
                        M_devctrl[dev_id, devctrl_id] += 1.0

    def _accumulate_routine_transitions(
        self,
        region: str,
        A_transition: np.ndarray,
    ) -> None:
        region_dir = os.path.join(self.data_dir, region)
        corpus_path = os.path.join(region_dir, "routine_device_corpus.txt")
        if not os.path.exists(corpus_path):
            return

        mappings = self.dictionary_manager.get_mappings(region)
        map_dev = mappings.get("device_dict", {})

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue

                global_seq = []
                for token in tokens:
                    try:
                        regional_id = int(token)
                    except ValueError:
                        continue

                    global_id = map_dev.get(regional_id)
                    if global_id is not None and global_id > 0:
                        global_seq.append(global_id)

                for i in range(len(global_seq) - 1):
                    src = global_seq[i]
                    dst = global_seq[i + 1]
                    A_transition[src, dst] += 1.0

    @staticmethod
    def _normalize_similarity(A: np.ndarray) -> np.ndarray:
        A = A.astype(np.float32, copy=False)
        np.fill_diagonal(A, 0.0)

        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return A / row_sums

    @staticmethod
    def _normalize_rows(A: np.ndarray) -> np.ndarray:
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return A / row_sums

    @staticmethod
    def _top_k_sparsify(A: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return A

        out = np.zeros_like(A)
        for i in range(A.shape[0]):
            row = A[i]
            if np.count_nonzero(row) <= k:
                out[i] = row
                continue

            top_idx = np.argpartition(row, -k)[-k:]
            out[i, top_idx] = row[top_idx]
        return out