import json
import torch
from torch.utils.data import Dataset

class bev_pretrain_dataset(Dataset):
    def __init__(self, scene_statements_path, bev_features_path):

        self.bev_features = [{"sample_token": key, "bev": tensor} 
                             for key, tensor in torch.load(bev_features_path).items()]
        
        with open(scene_statements_path, "r") as scene_statements_json:
            self.scene_statements = json.load(scene_statements_json)

        assert len(self.scene_statements) == len(self.bev_features), (f"Scene statement length ({len(self.scene_statements)}) 
                                                                      does not match BEV feature length ({len(self.bev_features)})")
        

    def __len__(self):
        return len(self.bev_features)

    def __getitem__(self, index):
        pass

