import os
import json
import glob

import torch
from torch.utils.data import Dataset


class bev_pretrain_dataset(Dataset):
    def __init__(self, bev_features_folder_path, scene_statements_path):
        self.nusc_samples = []
        
        self.bev_files_list = glob.glob(os.path.join(bev_features_folder_path, "*.pt"))
        
        with open(scene_statements_path, "r") as scene_statements_json:
            self.scene_statements = json.load(scene_statements_json)
        
    def get_scene_token(self, sample_token):
        for sample in self.nusc_samples:
            if sample["token"] == sample_token:
                return sample["scene_token"]
            
    def get_scene_statement(self, scene_token):
        for statement in self.scene_statements:
            if statement["scene_token"] == scene_token:
                return statement["statement"]
    
    def __len__(self):
        return len(self.bev_files_list)

    def __getitem__(self, index):
        
        # BEV -------------------------------------------
        bev_filename = self.bev_files_list[index]
        with open(bev_filename, "rb") as bev_file:
            bev = torch.load(bev_file)
        
        # Statement -------------------------------------
        sample_idx = bev_filename[:-3]
        scene_token = self.get_scene_token(sample_idx)
        statement = self.get_scene_statement(scene_token)

        return bev, statement
