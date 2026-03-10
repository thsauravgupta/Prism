import os
import glob
import importlib.util

class DictionaryManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.global_dicts = {
            'dayofweek_dict': {},
            'hour_dict': {},
            'device_dict': {},
            'device_control_dict': {}
        }
        
    def _load_module(self, filepath):
        module_name = "regional_dict"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def build_global_dictionaries(self):
        """Builds unified global dictionaries by scanning all regions."""
        regions = ['fr', 'kr', 'sp', 'us']
        for region in regions:
            dict_path = os.path.join(self.data_dir, region, "dictionary.py")
            if not os.path.exists(dict_path):
                continue
            
            module = self._load_module(dict_path)
            
            for dict_name in self.global_dicts.keys():
                regional_dict = getattr(module, dict_name, {})
                for key in regional_dict.keys():
                    if key not in self.global_dicts[dict_name]:
                        # Assign a new globally unique ID
                        new_id = len(self.global_dicts[dict_name])
                        self.global_dicts[dict_name][key] = new_id
                        
        # Ensure we have a default "Unknown" or "None" for devices if not present
        if 'None' not in self.global_dicts['device_dict']:
            self.global_dicts['device_dict']['None'] = len(self.global_dicts['device_dict'])
            
        return self.global_dicts

    def get_mappings(self, region):
        """Returns a mapping from regional ID to global ID for a given region."""
        dict_path = os.path.join(self.data_dir, region, "dictionary.py")
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary not found for region: {region}")
            
        module = self._load_module(dict_path)
        mappings = {}
        
        for dict_name in self.global_dicts.keys():
            regional_dict = getattr(module, dict_name, {})
            mapping = {}
            for k, regional_id in regional_dict.items():
                global_id = self.global_dicts[dict_name].get(k)
                mapping[regional_id] = global_id
            mappings[dict_name] = mapping
            
        return mappings

if __name__ == "__main__":
    manager = DictionaryManager()
    global_dicts = manager.build_global_dictionaries()
    print("Global Dictionary Sizes:")
    for name, d in global_dicts.items():
        print(f"  {name}: {len(d)} entries")
