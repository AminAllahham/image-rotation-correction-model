from tkinter import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from ConvNeuralNet import ConvNeuralNet
from DatasetLoader import DatasetLoader






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformations = transforms.Compose([transforms.ToTensor()])

train_set = DatasetLoader('training-data', 'training-dataset.csv', 224, 224,2,transformations)

testing_set = DatasetLoader('testing-data','testing-dataset.csv', 224, 224,2,transformations)

train_set_size = len(train_set)

batch_size = 64
learning_rate = 0.001
num_epochs = 10

train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)

testing_dataset_loader = torch.utils.data.DataLoader(dataset = testing_set, batch_size = batch_size,shuffle = True)

print("train_dataset_loader: ", len(train_dataset_loader))

model = ConvNeuralNet()



loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

total_step = len(train_dataset_loader)


losses = {}
losses['train'] = []
losses['test'] = []

acc = {}
acc['train'] = []
acc['test'] = []



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
                predicted = predicted.view(1,-1)
                
                labels = labels / 90
                train_total += labels.size(0)
                closeEnough = 0.5 / 90

                train_correct += (abs(predicted - labels) < closeEnough).float().sum().item()

            
                delta = abs(predicted - labels).sum().item() / labels.size(0)

                labels = labels.view(-1,1)
                loss = loss_function(output, labels)
                train_accuracy = (train_correct / train_total) * 100
                losses['train'].append(loss.item())
                acc['train'].append(train_correct / train_total)

            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Training Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item(), (train_correct / train_total) * 100))
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
                loss = loss_function(output, labels)


                losses['test'].append(loss.item())

                test_accuracy = (test_correct / test_total) * 100
                acc['test'].append(test_accuracy)

                valid_loss += loss.item()

                print('Testing Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item(), test_accuracy))

                # Save the model checkpoint
                torch.save(model.state_dict(), 'model.ckpt')





# load model and test with random image
model.load_state_dict(torch.load('model.ckpt'))
model.eval()
ImageForTest = 'testing-data/000000000000.jpg'
ImageForTest = Image.open(ImageForTest)
ImageForTest = ImageForTest.resize((224,224))

ImageForTest = transformations(ImageForTest)

ImageForTest = ImageForTest.unsqueeze(0)
ImageForTest = ImageForTest.to(device)

output = model(ImageForTest)

print(output)






