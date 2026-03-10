import os
import pickle
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, dictionary_manager, data_dir="data"):
        self.dictionary_manager = dictionary_manager
        self.data_dir = data_dir
        self.regions = ['fr', 'kr', 'sp', 'us']

    def load_and_preprocess(self):
        """Loads pkl files from all regions and converts them to global IDs."""
        logger.info("Building global dictionaries...")
        global_dicts = self.dictionary_manager.build_global_dictionaries()
        
        # We need mapping dictionaries for fast access
        global_dayofweek = global_dicts['dayofweek_dict']
        global_hour = global_dicts['hour_dict']
        global_device = global_dicts['device_dict']
        global_control = global_dicts['device_control_dict']
        
        # Determine the index of the "unknown/none" device to use as a fallback
        device_none_id = global_device.get('None', 0)
        
        all_X_trn, all_y_trn = [], []
        all_X_vld, all_y_vld = [], []
        all_X_tst, all_y_tst = [], []

        for region in self.regions:
            region_dir = os.path.join(self.data_dir, region)
            if not os.path.isdir(region_dir):
                continue
                
            logger.info(f"Processing region: {region}")
            mappings = self.dictionary_manager.get_mappings(region)
            
            # Helper to map a sequence array to global IDs
            def remap_data(data):
                if data is None or len(data) == 0:
                    return None
                
                # data shape is (instances, 10, 5) 
                # columns: dayofweek, hour, device, [unknown, device_control]
                mapped_data = np.zeros_like(data)
                
                map_dow = mappings.get('dayofweek_dict', {})
                map_hr = mappings.get('hour_dict', {})
                map_dev = mappings.get('device_dict', {})
                map_ctrl = mappings.get('device_control_dict', {})
                
                # Vectorized mapping using numpy advanced indexing or list comprehension
                # Since it might be slow with loops, we can use np.vectorize
                vector_dow = np.vectorize(lambda x: map_dow.get(x, 0))
                vector_hr = np.vectorize(lambda x: map_hr.get(x, 0))
                vector_dev = np.vectorize(lambda x: map_dev.get(x, device_none_id))
                vector_ctrl = np.vectorize(lambda x: map_ctrl.get(x, 0))
                
                mapped_data[:, :, 0] = vector_dow(data[:, :, 0])
                mapped_data[:, :, 1] = vector_hr(data[:, :, 1])
                mapped_data[:, :, 2] = vector_dev(data[:, :, 2])
                mapped_data[:, :, 3] = data[:, :, 3] # Keep unknown as is
                mapped_data[:, :, 4] = vector_ctrl(data[:, :, 4])
                
                return mapped_data

            # Split into sequences (9 steps) and target (10th step device)
            def process_split(filename):
                filepath = os.path.join(region_dir, filename)
                if not os.path.exists(filepath):
                    return None, None
                
                with open(filepath, 'rb') as f:
                    raw_data = pickle.load(f)
                    
                mapped = remap_data(raw_data)
                # Feature (context): First 9 events
                X = mapped[:, :-1, :]
                # Target: Predict the device type of the 10th event
                y = mapped[:, -1, 2] 
                return X, y

            x_trn, y_trn = process_split('trn_instance_10.pkl')
            if x_trn is not None:
                all_X_trn.append(x_trn)
                all_y_trn.append(y_trn)
                
            x_vld, y_vld = process_split('vld_instance_10.pkl')
            if x_vld is not None:
                all_X_vld.append(x_vld)
                all_y_vld.append(y_vld)
                
            x_tst, y_tst = process_split('test_instance_10.pkl')
            if x_tst is not None:
                all_X_tst.append(x_tst)
                all_y_tst.append(y_tst)

        # Concatenate all regions
        X_train = np.concatenate(all_X_trn, axis=0) if all_X_trn else np.array([])
        y_train = np.concatenate(all_y_trn, axis=0) if all_y_trn else np.array([])
        
        X_val = np.concatenate(all_X_vld, axis=0) if all_X_vld else np.array([])
        y_val = np.concatenate(all_y_vld, axis=0) if all_y_vld else np.array([])
        
        X_test = np.concatenate(all_X_tst, axis=0) if all_X_tst else np.array([])
        y_test = np.concatenate(all_y_tst, axis=0) if all_y_tst else np.array([])
        
        logger.info(f"Total training instances: {len(X_train)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), global_device
