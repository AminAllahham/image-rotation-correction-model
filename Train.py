# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ConvNeuralNet import ConvNeuralNet
from DatasetGenerator import RadomImagesRotationDatasetGenerator
from DatasetLoader import DatasetLoader

# Define relevant variables for the ML task
batch_size = 64
learning_rate = 0.001
num_epochs = 20


# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transformations = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose([ ])

train_set = DatasetLoader('./datasets/images-dataset.csv',224, 224,2,transformations)





train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,
                                           shuffle = True)

model = ConvNeuralNet()

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

total_step = len(train_dataset_loader)


for epoch in range(num_epochs):
	#Load in the data in batches using the train_loader object
   
    for i, (images, labels) in enumerate(train_dataset_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)

        print(labels)

        loss = criterion(outputs, labels)

        print(learning_rate)
        print(loss)


        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        else:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        

