import os
import json

import torch
from torch.utils.data import Dataset


class bev_drivelm_dataset(Dataset):
    def __init__(self, bev_folder_train, bev_folder_val, drivelm_json_path):

        self.bev_folder_train = bev_folder_train
        self.bev_folder_val = bev_folder_val

        with open(drivelm_json_path, "r") as drivelm_json:
            self.drivelm_dict = json.load(drivelm_json)

        self.drivelm = []
        for scene in self.drivelm_dict.values():
            for keyframe, keyframe_data in scene["key_frames"].items():
                for q_type, qs in keyframe_data["QA"].items():
                    for q_dict in qs:
                        question = q_dict["Q"]
                        answer = q_dict["A"]
                        self.drivelm.append({"sample_token": keyframe,
                                             "question": question,
                                             "answer": answer,
                                             "q_type": q_type})

    def __len__(self):
        return len(self.drivelm)

    def __getitem__(self, index):
        drivelm_item = self.drivelm[index]

        # BEV -------------------------------------------
        bev_filename = os.path.join(self.bev_folder_train,
                                    drivelm_item["sample_token"] + ".pt")

        if not os.path.exists(bev_filename):
            bev_filename = os.path.join(self.bev_folder_val,
                                        drivelm_item["sample_token"] + ".pt")

        with open(bev_filename, "rb") as bev_file:
            bev = torch.load(bev_file)
            bev = bev.squeeze()

        return {"bev": bev,
                "q": drivelm_item["question"],
                "a": drivelm_item["answer"],
                "q_type": drivelm_item["q_type"]}

