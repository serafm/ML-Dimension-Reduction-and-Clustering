import glob
import os
import warnings
from random import *
from sklearn.metrics import f1_score
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neural_network import MLPRegressor
from skmeans import SKMeans

K = []
K_before = []
count = 0
gray_img = []
categories = dict()

for i in range(10):
    n = randint(1, 4000)
    K.append(n)
    K_before.append(n)


for i in range(len(K)):
    openDir = os.listdir('train_data/' + str(K[i]))  # dir is your directory path
    number_files = len(openDir)  # number of images in dir
    while number_files < 50:
        K[i] = randint(1, 4000)
        while K[i] in K_before:
            K[i] = randint(1, 4000)
        K_before = K
        openDir = os.listdir('train_data/' + str(K[i]))  # dir is your directory path
        number_files = len(openDir)  # number of images in dir

print("DONE: Selected 10 random categories.")


# image to gray
def rgb2gray(rgb_img):
    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]
    gray = (0.299 * red + 0.587 * green + 0.114 * blue)
    return gray


# (1a) PCA algorithm
def PCA(X, num_components):

    # Subtract the mean of each variable
    X_meaned = X - np.mean(X, axis=0)

    # Calculate the Covariance Matrix
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Compute the Eigenvalues and Eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Sort Eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Select a subset from the rearranged Eigenvalue matrix
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced


real_labels = []
col = 0
gray_images_dataframe = pd.DataFrame()
print("Converting images from RGB to Gray")
# Open 50 images of 10 random categories and set their color to gray
for i in K:
    openDir = os.listdir('train_data/' + str(i))  # dir is your directory path
    number_files = len(openDir)  # number of images in dir
    if number_files >= 50:  # get the directories with 50 images or more so we can choose 50 of them
        for filename in glob.glob('train_data/' + str(i) + '/*.jpg'):  # for each image do:
            if count < 50:  # count to get only 50
                colourImg = Image.open(str(filename))
                colourPixels = colourImg.convert("RGB")
                colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
                gray_img = []

                # PCA
                for x in range(64):
                    for y in range(64):
                        red = colourArray[x][y][0]
                        green = colourArray[x][y][1]
                        blue = colourArray[x][y][2]
                        gray = 0.299 * red + 0.587 * green + 0.114 * blue
                        gray_img.append(gray)
                count += 1
                real_labels.append(i)
                gray_images_dataframe[col] = gray_img
                col += 1
    count = 0

print("DONE: Converted images to gray.")

print(gray_images_dataframe)
print("SIZE: ", len(gray_images_dataframe))


"""
# Load test_data
print("Loading test images")
test_data_dataframe = pd.DataFrame()
i = 1
for filename in glob.glob('test_data/*.jpg'):

    colourImg = Image.open(str(filename))
    colourPixels = colourImg.convert("RGB")
    colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
    gray_img = []

    for x in range(64):
        for y in range(64):
            red = colourArray[x][y][0]
            green = colourArray[x][y][1]
            blue = colourArray[x][y][2]
            gray = 0.299 * red + 0.587 * green + 0.114 * blue
            gray_img.append(gray)

    test_data_dataframe[i] = gray_img
    i += 1

print("DONE: Loaded test images.")
"""

"""
# Valid Data
print("Loading valid images")
valid_dataframe = pd.DataFrame()
num = 1
for i in K:
    for filename in glob.glob('val_data/' + str(i) + '/*.jpg'):
            colourImg = Image.open(str(filename))
            colourPixels = colourImg.convert("RGB")
            colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
            gray_img = []

            for x in range(64):
                for y in range(64):
                    red = colourArray[x][y][0]
                    green = colourArray[x][y][1]
                    blue = colourArray[x][y][2]
                    gray = 0.299 * red + 0.587 * green + 0.114 * blue
                    gray_img.append(gray)

            valid_dataframe[num] = gray_img
            num += 1
    num = 0

print("DONE: Loaded valid images.")
"""

# (1a) PCA Algorithm
images = gray_images_dataframe.iloc[:, :]
images = np.array(images)


#print("PCA started.")

# M=100
#reduced_images100 = PCA(images, 100)
#reduced_images100 = reduced_images100.transpose()

# M=50
reduced_images50 = PCA(images.transpose(), 50)
#reduced_images50 = reduced_images50.transpose()
print(reduced_images50)
print(len(reduced_images50))

# M=20
#reduced_images25 = PCA(images, 25)
#reduced_images25 = reduced_images25.transpose()

#print("DONE: PCA.")


"""
# (1b) Autoencoder
M = 50
d = 4096

print("Autoencoder started.")

autoencoder = MLPRegressor(hidden_layer_sizes=(d, d//4, M, d//4, d),
                           activation='tanh',
                           solver='adam',
                           learning_rate_init=0.0001,
                           max_iter=5,
                           tol=0.0000001,
                           verbose=True)


test_images = test_data_dataframe.iloc[:, :]
autoencoder.fit(images, test_images)
x_reconst = autoencoder.predict(images)

print("DONE: Autoencoder.")
"""


# (2a) K-Means algorithms

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

"""
# K-Means Euclidean Distance
print("Started K-Means Euclidean Distance")
kmeans = KMeans(init="random", n_clusters=10, n_init=10, max_iter=300, random_state=42)
predicted = kmeans.fit_predict(reduced_images50)

purityScore = purity_score(real_labels, predicted_labels)
print("Purity Score: ", purityScore)
"""

"""
# K-Means Cosine Distance
k_means_cosine = SKMeans(10, iters=15)
predicted = k_means_cosine.fit(reduced_images50)
print(predicted)
print(len(predicted))
purityScore = purity_score(labels, predicted)
print("Purity Score: ", purityScore)
"""


# (2b) agglomerative hierarchical clustering
agglomerative_hierarchical_clustering = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
agglomerative_hierarchical_clustering.fit_predict(reduced_images50)
labels = agglomerative_hierarchical_clustering.labels_
print(labels)