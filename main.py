import glob
import os
from random import *
from PIL import Image
import numpy as np


# Convert RGB values to grayscale values.
def rgb2gray_linear(rgb_img):

    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]

    gray_img = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue)

    return gray_img


K = []
count = 0
gray_img = []
categories = dict()

for i in range(10):
    K.append(randint(1, 4000))


# Open 50 images
for i in K:
    categories[i] = []
    list = os.listdir('train_data/' +str(i))  # dir is your directory path
    number_files = len(list) # number of images in dir
    if number_files >= 50:
        for filename in glob.glob('train_data/' +str(i) + '/*.jpg'):

            if count < 50:
                color_img = np.asarray(Image.open(str(filename))) / 255
                for x in range(64):
                    for y in range(64):
                        red = color_img[x][y][0]
                        green = color_img[x][y][1]
                        blue = color_img[x][y][2]
                        gray = 0.299*red + 0.587*green + 0.114*blue
                        gray_img.append(gray)
                categories[i].append(gray_img)
                gray_img = []
                count+=1

    count = 0


"""b = next(iter(categories))
print(len(categories[b]))"""


# PCA algorithm
#def PCA():


