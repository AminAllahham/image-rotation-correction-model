# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchmetrics import Accuracy

from ConvNeuralNet import ConvNeuralNet
from DatasetGenerator import RadomImagesRotationDatasetGenerator
from DatasetLoader import DatasetLoader

# Define relevant variables for the ML task
batch_size = 64
learning_rate = 0.001
num_epochs = 2


# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transformations = transforms.Compose([
    
        transforms.ToTensor(),
    ])

train_set = DatasetLoader('./datasets/images-dataset.csv',224, 224,2,transformations)





train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,
                                           shuffle = True)

model = ConvNeuralNet()

loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

total_step = len(train_dataset_loader)


for epoch in range(num_epochs):
	#Load in the data in batches using the train_loader object
    correct = 0
    total = 0
    for i, (image, label) in enumerate(train_dataset_loader):
        # Move tensors to the configured device
        image = image.to(device)
        label = label.to(device)
        
        output = model(image)
        
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).float().sum().item()

        



        label = label.view(-1,1)
        loss = loss_function(output, label)

       
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        






