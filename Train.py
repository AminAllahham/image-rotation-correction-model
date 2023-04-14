import os

import torch

from CnnNet import ConvNeuralNet
from DatasetLoader import DatasetLoader
from Transforms import transform
from Utils import (draw_compare_graph, get_average_values,
                   sinAndCosToRotationsDegrees)


device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

print('Using Gpu device:', device)


num_epochs = 16
batch_size = 8
learning_rate = 0.0001

max_valid_delta = 0.4


# 
number_of_rows_to_load = 1000


train_set = DatasetLoader('training-data/training-data.csv',1,transform, number_of_rows_to_load)
validation_set = DatasetLoader('training-data/validation-data.csv',1,transform, number_of_rows_to_load)

train_dataset_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size,shuffle = True)
validating_dataset_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)


print('Train Dataset Size: ', len(train_set))
print('Validation Dataset Size: ', len(validation_set))
     
model = ConvNeuralNet()
model = model.to(device)


criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if os.path.exists('model.ckpt'):
    model.load_state_dict(torch.load('model.ckpt'))

losses = { 'train': [], 'validate': [] }
accuracies = { 'train': [], 'validate': [] }

for epoch in range(num_epochs):
    losses['train'].append([])
    losses['validate'].append([])
    accuracies['train'].append([])
    accuracies['validate'].append([])

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


            predicted = sinAndCosToRotationsDegrees(outputs.cpu().data)
            target = sinAndCosToRotationsDegrees(labels.cpu().data)

            delta = abs(predicted - target)

            print('DELTA: ', delta)

                
            total_correct += (delta < max_valid_delta).sum().item()

            epoch_acc = 100 * (total_correct / total_running)

            if phase == 'train':
                losses['train'][epoch].append(loss.item())
                accuracies['train'][epoch].append(epoch_acc)
                print('Training Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%' .format(epoch + 1, num_epochs, i + 1, len(train_dataset_loader), loss.item(), epoch_acc))
            else:
                losses['validate'][epoch].append(loss.item())
                accuracies['validate'][epoch].append(epoch_acc)
                print('Testing Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%' .format(epoch + 1, num_epochs, i + 1, len(validating_dataset_loader), loss.item(), epoch_acc))
                torch.save(model.state_dict(), 'model.ckpt')
                torch.save(model, 'model.pth')

      
draw_compare_graph(get_average_values(losses['train']),get_average_values(losses['validate']), 'Loss', f'last-Loss.png')
draw_compare_graph(get_average_values(accuracies['train']), get_average_values(accuracies['validate']), 'Accuracy', f'last-Accuracy.png')



print('Finished Training 🚀')
print('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
torch.save(model, 'model.pth')
