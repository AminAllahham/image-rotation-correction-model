import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from DatasetLoader import DatasetLoader
from Utils import rotate_image, sinAndCosToRotationsDegrees

# load the model from disk and predict the output for the given image
model = torch.load('model.pth')
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# transform the image
transform =transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



testing_dataset = DatasetLoader('training-data','testing-data.csv',2,transform)
testing_dataset_loader = torch.utils.data.DataLoader(dataset = testing_dataset, batch_size = 1,shuffle = True)

# load 25 random images from the validation set
# accuracy = 0
# for i in range(25):
#     img, label = testing_dataset[i]
#     img = img.unsqueeze(0)
#     img = img.to(device)
    
#     output = model(img)
#     output = sinAndCosToRotationsDegrees(output.data).item()

#     label = np.arctan2(label[0], label[1])
#     label = np.degrees(label)

#     print([output, label.item()])
#     delta = abs(output - label).item()

#     if delta < 0.7:
#         accuracy += 1
#     else:
#         print("DELTA ON Wrong: ", delta)
    

# print('Accuracy on 25 Test Image: ', (accuracy/25) * 100 , '%')

       


# create output folder if not exists
if not os.path.exists('output'):
    os.makedirs('output')
    
images = os.listdir('images')

for image in images:
    img_path = 'images/' + image
    img = cv2.imread(img_path)
    copyImage = cv2.imread(img_path)

    img = Image.fromarray(img)
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    output = sinAndCosToRotationsDegrees(output.data).item()
    print(output)

    rotatedImage = rotate_image(copyImage, -output)
    mergedImage = np.concatenate((copyImage, rotatedImage), axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(mergedImage, 'Before', (20, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(mergedImage, 'After', (copyImage.shape[1] + 20, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    mergedImage = Image.fromarray(mergedImage)
    mergedImage.save('output/' + image)






    

    










    
    
