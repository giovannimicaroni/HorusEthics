import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class RFW_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)