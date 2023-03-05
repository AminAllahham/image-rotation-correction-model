
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from ConvNeuralNet import ConvNeuralNet
from DatasetLoader import DatasetLoader


def sinAndCosToDegree(values):
    values = np.asarray(values)
    values = np.arctan2(values[:, 0], values[:, 1])
    values = np.degrees(values)
    return values



def calculate_accuracy(outputs, labels , train_total = 0, train_correct = 0):
    predictedAngles = sinAndCosToDegree(outputs.data)
    labelAngles = sinAndCosToDegree(labels)

    print('Predicted Angles:',predictedAngles)
    print('Label Angles:',labelAngles)

    closeEnough = 0.5

    train_total += labels.size(0)
    train_correct += (abs(predictedAngles - labelAngles) < closeEnough).sum().item()

    accuracy = train_correct / train_total

    return accuracy



batch_size = 8
num_epochs = 16
learning_rate = 0.001


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformations = transforms.Compose([transforms.ToTensor()])

train_set = DatasetLoader('training-data', 'training-dataset.csv', 224, 224,2,transformations)

validation_set = DatasetLoader('validation-data','validation-dataset.csv', 224, 224,2,transformations)

train_set_size = len(train_set)



train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)

validating_dataset_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size,shuffle = True)
       
model =  ConvNeuralNet()
model = model.to(device)

# Load the model from the checkpoint
model.load_state_dict(torch.load('model.ckpt'))

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


    for phase in ['train', 'validate']:
        if phase == 'train':
            model.train()
            for i, (images, labels) in enumerate(train_dataset_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)


                acc =  calculate_accuracy(outputs, labels, train_total, train_correct)

            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Trining Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), acc * 100))

        else:
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(validating_dataset_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss = loss_function(outputs, labels)
                    calculate_accuracy(outputs, labels, test_total, test_correct)

                    torch.save(model.state_dict(), 'model.ckpt')

                    print('Validation Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), acc * 100))

torch.save(model, 'model.pth')






