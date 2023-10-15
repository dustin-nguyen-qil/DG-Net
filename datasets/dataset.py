from typing import Any
import torch
import pickle
import random
from torch.utils.data import Dataset
from PIL import Image

class VidDataset(Dataset):
    def __init__(self, data_path, transform) -> None:
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.data = self.get_data(self.data_path)

    def get_data(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)['data']
        return data # list of tracklets
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        tracklet = self.data[index]
        p_id = tracklet['p_id']
        cam_id = tracklet['cam_id']
        clothes_id = tracklet['clothes_id']
        img_paths = tracklet['img_paths']
        imgs = []
        for img_path in img_paths:
            img = Image.open(img_path)
            img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        return torch.cat(imgs, 0), img_paths, p_id, cam_id, clothes_id
    
# for each tracklet_a, pick 5 ids, then for each id pick 1 clothes_id

def get_dataset_b(data_path, transform):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)['data']
        
    
    



    
    
        