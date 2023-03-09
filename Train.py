
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from ConvNeuralNet import ConvNeuralNet


from DatasetLoader import DatasetLoader


def sinAndCosToDegree(values):
    values = np.asarray(values)
    values = np.arctan2(values[:, 0], values[:, 1])
    values = np.degrees(values)
    return values 



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
# model.load_state_dict(torch.load('model.ckpt'))

loss_function =  nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


losses = {'train': [], 'validate': []}
accuracies = {'train': [], 'validate': []}


total_train_step = len(train_dataset_loader)
total_validate_step = len(validating_dataset_loader)

print('Total Training Step: ', total_train_step)
print('Total Validation Step: ', total_validate_step)

total_step = total_train_step + total_validate_step

closeEnough = 0.5

    

for epoch in range(num_epochs):
    train_total = 0
    train_correct = 0

    validate_total = 0
    validate_correct = 0

    for phase in ['train', 'validate']:
        if phase == 'train':
            model.train()
            for i, (images, labels) in enumerate(train_dataset_loader):
                images = images.to(device)
                labels = labels.to(device)

                print(images.size())
                print(labels.size())

                outputs = model(images)
                loss = loss_function(outputs, labels)


                predicted_angles = sinAndCosToDegree(outputs.data)
                labels_angles = sinAndCosToDegree(labels)


    

                train_total += labels.size(0)
                train_correct += (abs(predicted_angles - labels_angles) < closeEnough).sum().item()
                train_accuracy = train_correct / train_total


                losses['train'].append(loss.item())
                accuracies['train'].append(train_accuracy)
                

            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Trining Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), train_accuracy * 100))

        else:
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(validating_dataset_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss = loss_function(outputs, labels)

                    predicted_angles = sinAndCosToDegree(outputs.data)
                    labels_angles = sinAndCosToDegree(labels)

                    validate_total += labels.size(0)
                    validate_correct += (abs(predicted_angles - labels_angles) < closeEnough).sum().item()

                    validate_accuracy = validate_correct / validate_total

                    losses['validate'].append(loss.item())
                    accuracies['validate'].append(validate_accuracy)

                    torch.save(model.state_dict(), 'model.ckpt')

                    print('Validation Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), validate_accuracy * 100))

torch.save(model, 'model.pth')
# 582 712


# save loss graph png
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validate'], label='Validation loss')
plt.legend(frameon=False)
plt.savefig('loss.png')

# save accuracy graph png
plt.plot(accuracies['train'], label='Training accuracy')
plt.plot(accuracies['validate'], label='Validation accuracy')
plt.legend(frameon=False)
plt.savefig('accuracy.png')






