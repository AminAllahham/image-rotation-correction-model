import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Creating a CNN class
# Creating a CNN class
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        # image size = 224 x 224
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # [(height or width) - kernel size + 2*padding] / stride + 1
        self.fc1 =  nn.Linear(64 * 56 * 56, 1)

    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)

        
        print(out.shape)
       
        return out






# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms


# # Creating a CNN class
# # Creating a CNN class
# class ConvNeuralNet(nn.Module):
# 	#  Determine what layers and their order in CNN object 
#     # def __init__(self):
#     #     super(ConvNeuralNet, self).__init__()
#     #     # image size = 224 x 224
#     #     self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
#     #     self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
#     #     self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
#     #     self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
#     #     self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
#     #     self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        

#     #     # image size = 224 x 224
#     #     # RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x179776 and 46656x2)
#     #     self.fc1 = nn.Linear(64 * 56 * 56, 1)
        
    
#     # # Progresses data across layers    
#     # def forward(self, x):
#     #     out = self.conv_layer1(x)
#     #     out = self.conv_layer2(out)
#     #     out = self.max_pool1(out)
        
#     #     out = self.conv_layer3(out)
#     #     out = self.conv_layer4(out)
#     #     out = self.max_pool2(out)
                
#     #     out = out.reshape(out.size(0), -1)
        
#     #     out = self.fc1(out)
#     #     print(out)
       
#     #     return out
#     def __init__(self):
#         super(ConvNeuralNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.fc = nn.Linear(64 * 56 * 56, 1)

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv2(self.conv1(x))))
#         x = self.pool(nn.functional.relu(self.conv4(self.conv3(x))))
#         x = x.view(-1, 64 * 56 * 56)
#         print(x.size(0))
#         # x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#         return x
