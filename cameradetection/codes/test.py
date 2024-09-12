import numpy as np
from PIL import Image
import os
import random
import pandas as pd

IMG_SIZE = 1024
DATASET = 'RFW'


if __name__ == '__main__':
    tp = [0.9829, 0.9355, 0.9672, 0.9756, 0.9591, 0.9587, 0.9820, 0.9657, 0.9678, 0.9668, 0.9497, 0.9812, 0.9797, 0.9745, 0.9614, 0.9587, 0.9822, 0.9644, 0.9462, 0.9727]
    tn = [0.7825, 0.7861, 0.7209, 0.7562, 0.7650, 0.7263, 0.7518, 0.7845, 0.7848, 0.7681, 0.6751, 0.7580, 0.7532, 0.7543, 0.7846, 0.6225, 0.7351, 0.7297, 0.8153, 0.7682]
    tp_array = np.array(tp)
    tn_array = np.array(tn)
    np.save('partialTP', tp_array)
    np.save('partialTN', tn_array)

