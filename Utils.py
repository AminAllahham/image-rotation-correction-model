
import cv2
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


