import os

import numpy as np
import torch
import torchvision.transforms as transforms

from CnnNet import ConvNeuralNet
from DatasetLoader import DatasetLoader
from Utils import sinAndCosToRotationsDegrees

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_epochs = 10
batch_size = 8
learning_rate = 0.0001

max_valid_delta = 0.5

transform =transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_set = DatasetLoader('training-data', 'training-data.csv',2,transform)
validation_set = DatasetLoader('training-data','validation-data.csv',2,transform)

train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)
validating_dataset_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)


print('Train Dataset Size: ', len(train_set))
print('Validation Dataset Size: ', len(validation_set))
     
model = ConvNeuralNet()
model = model.to(device)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if os.path.exists('model.ckpt'):
    model.load_state_dict(torch.load('model.ckpt'))


for epoch in range(num_epochs):
    for phase in ['train', "validate"]:
        if phase == 'train':
            model.train()
            dataset = train_dataset_loader
        else:
            model.eval()
            dataset = validating_dataset_loader

        total_running = 0
        total_correct = 0


        for i, (images, labels) in enumerate(dataset):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()


            total_running += labels.size(0)


            predicted = sinAndCosToRotationsDegrees(outputs.data)
            target = sinAndCosToRotationsDegrees(labels)

            delta = abs(predicted - target)

            print('DELTA: ', delta)

                
            total_correct += (delta < max_valid_delta).sum().item()

            epoch_acc = 100 * (total_correct / total_running)

            if phase == 'train':
                print('Training Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%' .format(epoch + 1, num_epochs, i + 1, len(train_dataset_loader), loss.item(), epoch_acc))
            else:
                print('Testing Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%' .format(epoch + 1, num_epochs, i + 1, len(validating_dataset_loader), loss.item(), epoch_acc))
                torch.save(model.state_dict(), 'model.ckpt')
                torch.save(model, 'model.pth')

                

print('Finished Training 🚀')
# Last Number of parameters:  
print('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
torch.save(model, 'model.pth')






