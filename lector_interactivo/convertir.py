import numpy as np
from red2 import Sample
from PIL import Image

def img_to_array(filename):
    img = Image.open(filename)
    data = np.array(img)
    return data

def array_to_sample(array_img):
    list_img = []
    for row in array_img:
        for slot in row:
            if slot[0] == 255:
                list_img += [0.01]
            else:
                list_img += [1]
    return Sample(list_img)
