import numpy as np
import torch
import torchvision.transforms as transforms

from CnnNet import ConvNeuralNet
from DatasetLoader import DatasetLoader
from Utils import sinAndCosToRotationsDegrees

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper Parameters
num_epochs = 10
batch_size = 1
learning_rate = 0.001


# Dataset transform & Loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = DatasetLoader('training-data', 'training-data.csv', 224, 224,2,transform)
validation_set = DatasetLoader('training-data','validation-data.csv', 224, 224,2,transform)

train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)
validating_dataset_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size,shuffle = True)


# Model
model = ConvNeuralNet()
model = model.to(device)


# Loss and Optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


phases = ['train', 'validate']

for epoch in range(num_epochs):
    

    for phase in phases:
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

            total_correct += (delta < 0.5).sum().item()

            epoch_acc = total_correct / total_running * 100

            if phase == 'train':
                print('Train Loss: {:.4f} Acc: {:.4f}'.format(loss, epoch_acc))
            else:
                torch.save(model.state_dict(), 'model.ckpt')
                print('Validation Loss: {:.4f} Acc: {:.4f}'.format(loss, epoch_acc))


torch.save(model, 'model.pth')

            

            
  


    

          
           
           




            


            

          

    
