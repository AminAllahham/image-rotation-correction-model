import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# load the model from disk and predict the output for the given image
model = torch.load('model.pth')
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the image
img = Image.open('test.jpg')

img = img.resize((224,224), Image.ANTIALIAS)
# transform the image
transform =transforms.Compose([transforms.ToTensor()])
img = transform(img)

# add a batch dimension
img = img.unsqueeze(0)

# move the image to the device
img = img.to(device)

# predict the output

def sinAndCosToDegree(values):
    values = np.asarray(values)
    values = np.arctan2(values[:, 0], values[:, 1])
    values = np.degrees(values)
    return values

def rotate_image(image, angle): 
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result


with torch.no_grad():
    output = model(img)
    print('output',sinAndCosToDegree(output.numpy()))
    predictedAngle =  sinAndCosToDegree(output.numpy()).item()
    copyImage = cv2.imread('test.jpg')

    print('Predicted Angle: ', predictedAngle)
    rotatedImage = rotate_image(copyImage, -predictedAngle)
    cv2.imwrite('rotated.jpg', rotatedImage)
    

    
