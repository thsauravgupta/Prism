import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

class KGBuilder:
    def __init__(self, dictionary_manager, data_dir="data"):
        self.dictionary_manager = dictionary_manager
        self.data_dir = data_dir
        self.regions = ['fr', 'kr', 'sp', 'us']

    def build_adjacency_matrix(self):
        """
        Reads user routine corpora across regions and builds an adjacency matrix based on device transitions.
        """
        global_dicts = self.dictionary_manager.global_dicts
        if not global_dicts['device_dict']:
            self.dictionary_manager.build_global_dictionaries()
            
        num_devices = len(global_dicts['device_dict'])
        adj_matrix = np.zeros((num_devices, num_devices), dtype=np.float32)

        for region in self.regions:
            region_dir = os.path.join(self.data_dir, region)
            corpus_path = os.path.join(region_dir, 'routine_device_corpus.txt')
            if not os.path.exists(corpus_path):
                continue
                
            mappings = self.dictionary_manager.get_mappings(region)
            map_dev = mappings.get('device_dict', {})
            
            with open(corpus_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) < 2:
                        continue
                        
                    # Map tokens to global device IDs
                    global_seq = []
                    for token in tokens:
                        try:
                            # Tokens in the txt are strings representing original IDs
                            regional_id = int(token)
                            global_id = map_dev.get(regional_id, 0)
                            global_seq.append(global_id)
                        except ValueError:
                            pass
                            
                    # Build transition counts
                    for i in range(len(global_seq) - 1):
                        src = global_seq[i]
                        dst = global_seq[i + 1]
                        adj_matrix[src, dst] += 1.0

        # Normalize the adjacency matrix (row-wise) to get transition probabilities
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        adj_matrix_normalized = adj_matrix / row_sums
        
        # Add self connections
        np.fill_diagonal(adj_matrix_normalized, 1.0)

        logger.info(f"Generated KG Adjacency Matrix with shape: {adj_matrix_normalized.shape}")
        return adj_matrix_normalized
