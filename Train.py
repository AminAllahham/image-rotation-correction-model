import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tkinter import Image
import torch
import torch.nn as nn
import torch.optim as optim
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
    pretrained=True,
)

model.classifier[1] = nn.Linear(1280, 2)



model = model.to(device)


loss_function =  nn.MSELoss()

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
            print("Training...")
            for i, (images, labels) in enumerate(train_dataset_loader):
                images = images.to(device)
                labels = labels.to(device)
            
                output = model(images)

                predicted = output.data
                
            
                predictedAngles = sinAndCosToDegree(predicted)


                labelAngles = sinAndCosToDegree(labels)

                print(
                    'Predicted: ', predictedAngles,
                    '\n',
                    'Labels: ', labelAngles

                )

                delta = abs(predictedAngles - labelAngles) # numpy.ndarray
 
                deltaCloseEnough = 0.5

                train_total += labels.size(0)

                train_correct += (delta < deltaCloseEnough).sum().item()

                

                loss = loss_function(predicted, labels)
                loss.requires_grad = True 

                print('Loss: ', loss)

                losses['train'].append(loss.item())
                acc['train'].append(train_correct / train_total)

                accuracy = train_correct / train_total

        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), accuracy*100))


             
        else:
            print("Testing...")
            model.eval()
            valid_loss = 0.0
            for i, (images, labels) in enumerate(testing_dataset_loader):
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                predicted = output.data
                predicted = predicted.view(1,-1)
                
                labels = labels / 90
                test_total += labels.size(0)
                closeEnough = 0.5 / 90

                test_correct += (abs(predicted - labels) < closeEnough).float().sum().item()

                delta = abs(predicted - labels).sum().item() / labels.size(0)

                labels = labels.view(-1,1)
               




