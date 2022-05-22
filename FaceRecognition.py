import glob
import os
from random import *
from PIL import Image
import numpy as np


K = []
count = 0
gray_img = []
categories = dict()

for i in range(10):
    K.append(randint(1, 4000))


# Open 50 images of 10 random categories and set their color to gray
for i in K:
    categories[i] = []  # dictionary includes 50 images arrays of a category(person), Key:category Value: 50 arrays
    openDir = os.listdir('train_data/' + str(i))  # dir is your directory path
    number_files = len(openDir)  # number of images in dir
    if number_files >= 50:  # get the directories with 50 images or more so we can choose 50 of them
        for filename in glob.glob('train_data/' + str(i) + '/*.jpg'):  # for each image do:
            if count < 50:  # count to get only 50
                color_img = np.asarray(Image.open(str(filename))) / 255  # image to array of RGB
                for x in range(64):
                    for y in range(64):
                        red = color_img[x][y][0]
                        green = color_img[x][y][1]
                        blue = color_img[x][y][2]
                        gray = 0.299*red + 0.587*green + 0.114*blue
                        gray_img.append(gray)
                categories[i].append(gray_img)  # add the gray image to the dictionary
                gray_img = []  # initialise array as empty
                count += 1
    count = 0


"""b = next(iter(categories))
print(len(categories[b]))"""


# PCA algorithm
# def PCA():


