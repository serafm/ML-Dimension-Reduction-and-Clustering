import glob
import os
import warnings
from random import *
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neural_network import MLPRegressor
from skmeans import SKMeans

K = []
count = 0
gray_img = []
categories = dict()

for i in range(10):
    K.append(randint(1, 4000))


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

PCA_reduced_categories100 = dict()
PCA_reduced_categories50 = dict()
PCA_reduced_categories25 = dict()

gray_images_pca = dict()

# Open 50 images of 10 random categories and set their color to gray
for i in K:

    categories[i] = []  # dictionary includes 50 images arrays of a category(person), Key:category Value: 50 arrays
    gray_images_pca[i] = []
    openDir = os.listdir('train_data/' + str(i))  # dir is your directory path
    number_files = len(openDir)  # number of images in dir

    if number_files >= 50:  # get the directories with 50 images or more so we can choose 50 of them
        for filename in glob.glob('train_data/' + str(i) + '/*.jpg'):  # for each image do:
            if count < 50:  # count to get only 50
                color_img = np.asarray(Image.open(str(filename))) / 255  # image to array of RGB
                gray_image = rgb2gray(color_img)
                categories[i].append(gray_image)  # add the gray image to the dictionary

                # For PCA
                for x in range(64):
                    for y in range(64):
                        red = color_img[x][y][0]
                        green = color_img[x][y][1]
                        blue = color_img[x][y][2]
                        gray = 0.299 * red + 0.587 * green + 0.114 * blue
                        gray_img.append(gray)
                count += 1

                gray_images_pca[i].append(gray_img)
                gray_img = []
    count = 0



for key in categories:

    #PCA_reduced_categories100[key] = []
    #PCA_reduced_categories50[key] = []
    #PCA_reduced_categories25[key] = []

    images = gray_images_pca[key]

    reduced_image100 = PCA(images, 100)
    #reduced_image50 = PCA(images, 50)
    #reduced_image25 = PCA(images, 25)

    PCA_reduced_categories100[key] = reduced_image100
    #PCA_reduced_categories50[key].append(reduced_image50)
    #PCA_reduced_categories25[key].append(reduced_image25)
    break


# (1b) Autoencoder
M = 100
d = 4096

autoencoder = MLPRegressor(hidden_layer_sizes=(d, d//4, M, d//4, d),
                           activation='tanh',
                           solver='adam',
                           learning_rate_init=0.0001,
                           max_iter=20,
                           tol=0.0000001,
                           verbose=True)


"""
for key in categories:
    gray_image = categories[key][7]
    autoencoder.fit(gray_image, gray_image)
    x_reconst = autoencoder.predict(gray_image.reshape(-1, 64))
    break
"""


"""
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(gray_image.reshape(64, 64), 'gray')
plt.title('Input Image', fontsize=15)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(x_reconst.reshape(64, 64), 'gray')
plt.title('Reconstructed Image', fontsize=15)
plt.xticks([])
plt.yticks([])
plt.show()
"""


# (2a) K-Means algorithms

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# K-Means Euclidean Distance
for i in PCA_reduced_categories100:
    k_means = KMeans(init="random", n_clusters=10, n_init=10, max_iter=300, random_state=42)
    # fit_images = k_means.fit(PCA_reduced_categories100[i])
    predicted = k_means.fit_predict(PCA_reduced_categories100[i])

    purityScore = purity_score(PCA_reduced_categories100[i], predicted)
    print("Purity Score: ", purityScore)
    break


# K-Means Cosine Distance
"""
for i in PCA_reduced_categories100:
    k_means_cosine = SKMeans(10, iters=15)
    k_means_cosine.fit(PCA_reduced_categories100[i])
"""


"""
# (2b) agglomerative hierarchical clustering
for i in PCA_reduced_categories100:
    agglomerative_hierarchical_clustering = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
    agglomerative_hierarchical_clustering.fit(PCA_reduced_categories100[i])
    labels = agglomerative_hierarchical_clustering.labels_
"""