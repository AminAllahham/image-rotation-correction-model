import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom datasets


class DatasetLoader(Dataset):
    def __init__(self,datasetFolder, csv_path, labelIndex, transform=None):

        self.data = pd.read_csv(os.path.join(datasetFolder, csv_path))
        self.labelIndex = labelIndex
        self.labels = np.asarray(self.data.iloc[:, self.labelIndex])

        self.transform = transform
        self.datasetFolder = datasetFolder
    

    def __getitem__(self, index):
        label = self.labels[index]

        image = Image.open(os.path.join(self.data.iloc[index][1]))

        image = image.convert('RGB')

        
        label = torch.tensor(label)

        label = np.radians(label)

        label = torch.tensor([np.sin(label), np.cos(label)])

    
        if self.transform is not None:
            img_as_tensor = self.transform(image)
    
        
        return (img_as_tensor, label)

    def __len__(self):
        return len(self.data.index)





