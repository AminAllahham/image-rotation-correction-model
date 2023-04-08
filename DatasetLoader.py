import os

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms


class DatasetLoader(Dataset):
    def __init__(self, csv_path, labelIndex, transform=None):

        self.data = pd.read_csv(csv_path)
        self.labelIndex = labelIndex
        self.labels = np.asarray(self.data.iloc[:, self.labelIndex])

        self.transform = transform
    

    def __getitem__(self, index):
        label = self.labels[index]

        image = Image.open(os.path.join(self.data.iloc[index][0]))

        image = image.convert('L')

    

        label = torch.tensor([np.sin(np.radians(label)), np.cos(np.radians(label))])
    

        img_as_tensor = self.transform(image)            
    
        
        return (img_as_tensor, label)

    def __len__(self):
        return len(self.data.index)





