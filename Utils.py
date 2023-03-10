import math
import numpy as np

import torch


def sinAndCosToRotationsDegrees(values):
    values = np.asarray(values)
    values = np.arctan2(values[:, 0], values[:, 1])
    values = np.degrees(values)
    return values 
