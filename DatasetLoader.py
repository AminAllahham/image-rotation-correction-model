import os

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms


class DatasetLoader(Dataset):
    def __init__(self, csv_path, height, width, labelIndex, transform=None):

        self.data = pd.read_csv(csv_path)
        self.labelIndex = labelIndex
        self.labels = np.asarray(self.data.iloc[:, self.labelIndex])
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]


        image = Image.open(self.data.iloc[index][1])

        image = image.resize((self.height,self.width), PIL.Image.ANTIALIAS)

    
        image = image.convert('RGB')

        y_label = torch.tensor(single_image_label, dtype=torch.long)

        
        if self.transform is not None:
            img_as_tensor = self.transform(image)
    

        return (img_as_tensor, y_label)

    def __len__(self):
        return len(self.data.index)






