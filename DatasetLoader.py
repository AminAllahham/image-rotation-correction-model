import os

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms


class DatasetLoader(Dataset):
    def __init__(self,datasetFolder, csv_path,  height, width, labelIndex, transform=None):

        self.data = pd.read_csv(os.path.join(datasetFolder, csv_path))
        self.labelIndex = labelIndex
        self.labels = np.asarray(self.data.iloc[:, self.labelIndex])
        self.height = height
        self.width = width
        self.transform = transform
        self.datasetFolder = datasetFolder
    

    def __getitem__(self, index):
        single_image_label = self.labels[index]

        image = Image.open(os.path.join(self.data.iloc[index][1]))

        image = image.resize((self.height,self.width), PIL.Image.ANTIALIAS)

        image = image.convert('RGB')


        label = torch.tensor(single_image_label)

        label = np.radians(label)

        label = torch.tensor([np.sin(label), np.cos(label)])

    
        if self.transform is not None:
            img_as_tensor = self.transform(image)
    
        
        return (img_as_tensor, label)

    def __len__(self):
        return len(self.data.index)





