import math
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np


def sinAndCosToRotationsDegrees(values):
    values = np.asarray(values)
    values = np.arctan2(values[:, 0], values[:, 1])
    values = np.degrees(values)
    return values 


def rotate_image(image, angle): 
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result


def draw_compare_graph(x_values, y_values, title ,filename):
    fig = plt.figure(figsize=(8, 4))
    
    plt.plot(x_values, label='train')
    plt.plot(y_values, label='valid')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(filename)
    
    return os.path.abspath(filename)


def get_average_values(list):
    result = []
    for i in range(len(list)):
        result.append(np.average(list[i]))
          
    return result




def show_compare_graph(x_values, y_values, title ):
    fig = plt.figure(figsize=(8, 4))
    
    plt.plot(x_values, label='train')
    plt.plot(y_values, label='valid')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()