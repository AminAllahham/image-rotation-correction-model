import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from ConvNeuralNet import ConvNeuralNet
from DatasetLoader import DatasetLoader

batch_size = 64
learning_rate = 0.001
num_epochs = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transformations = transforms.Compose([transforms.ToTensor()])

train_set = DatasetLoader('./datasets/images-dataset.csv',224, 224,2,transformations)

# testing_set = DatasetLoader('./datasets/images-dataset-test.csv',224, 224,2,transformations)



train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)

# testing_dataset_loader = torch.utils.data.DataLoader(dataset = testing_set, batch_size = batch_size,shuffle = True)

model = ConvNeuralNet()

loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

total_step = len(train_dataset_loader)



for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (image, labels) in enumerate(train_dataset_loader):
        image = image.to(device)
        labels = labels.to(device)
    
        output = model(image)

        predicted = output.data
        predicted = predicted.view(1,-1)
        
        labels = labels / 90
        total += labels.size(0)
        closeEnough = 0.5 / 90

        print("predicted: ", predicted)
        print("label: ", labels)

        correct += (abs(predicted - labels) < closeEnough).float().sum().item()

    
        delta = abs(predicted - labels).sum().item() / labels.size(0)

        print("predicted: ", predicted)

        labels = labels.view(-1,1)
        loss = loss_function(output, labels)

       
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))




