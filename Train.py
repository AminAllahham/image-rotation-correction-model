from tkinter import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from DatasetLoader import DatasetLoader


def sinAndCosToDegree(values):
    values = np.asarray(values)
    values = np.arctan2(values[:, 0], values[:, 1])
    values = np.degrees(values)
    return values


batch_size = 16
num_epochs = 10
learning_rate = 0.001


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformations = transforms.Compose([transforms.ToTensor()])

train_set = DatasetLoader('training-data', 'training-dataset.csv', 224, 224,2,transformations)

testing_set = DatasetLoader('testing-data','testing-dataset.csv', 224, 224,2,transformations)

train_set_size = len(train_set)



train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)

testing_dataset_loader = torch.utils.data.DataLoader(dataset = testing_set, batch_size = batch_size,shuffle = True)
       
model = models.mobilenet_v2(
    pretrained=False,
)

model.classifier[1] = nn.Linear(1280, 2)

model = model.to(device)


loss_function =  nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



losses = {'train': [], 'test': []}
acc = {'train': [], 'test': []}


total_step = len(train_dataset_loader)

for epoch in range(num_epochs):
    train_total = 0
    train_correct = 0

    test_total = 0
    test_correct = 0


    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
            for i, (images, labels) in enumerate(train_dataset_loader):
                images = images.to(device)
                labels = labels.to(device)
            
                outputs = model(images)

                loss = loss_function(outputs, labels)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            
                predictedAngles = sinAndCosToDegree(outputs.data)
                labelAngles = sinAndCosToDegree(labels)

                closeEnough = 0.5

                train_total += labels.size(0)
                train_correct += (abs(predictedAngles - labelAngles) < closeEnough).sum().item()

                accuracy = train_correct / train_total

                
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), accuracy*100))
             
        else:
            model.eval()
            with torch.no_grad():
                for images, labels in testing_dataset_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)

                    loss = loss_function(outputs, labels)

                    predictedAngles = sinAndCosToDegree(outputs.data)
                    labelAngles = sinAndCosToDegree(labels)

                    closeEnough = 0.5

                    test_total += labels.size(0)
                    test_correct += (abs(predictedAngles - labelAngles) < closeEnough).sum().item()

                    accuracy = test_correct / test_total

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item(), accuracy*100))
            
            # Save the model checkpoint
            torch.save(model.state_dict(), 'model.ckpt')






