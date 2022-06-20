import glob
import os
from random import *
from pyclustering.utils import distance_metric, type_metric
from sklearn.metrics import f1_score
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import metrics
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from sklearn.neural_network import MLPRegressor

K = []
K_before = []
count = 0
gray_img = []
categories = dict()

# Select 10 random categories
for i in range(10):
    n = randint(1, 4000)
    K.append(n)
    K_before.append(n)

"""
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
"""


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
lbl = 0
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
                real_labels.append(lbl)
                gray_images_dataframe[col] = gray_img
                col += 1
    lbl += 1
    count = 0

print("DONE: Converted images to gray.")
real_labels = np.array(real_labels)
print(gray_images_dataframe)
print("SIZE: ", len(gray_images_dataframe))


# (1a) PCA Algorithm
images = gray_images_dataframe.iloc[:, :]
images = np.array(images)

print("PCA started.")

# M=100
reduced_images100 = PCA(images.transpose(), 100)
reduced_images100 = reduced_images100

# M=50
reduced_images50 = PCA(images.transpose(), 50)
reduced_images50 = reduced_images50

# M=20
reduced_images25 = PCA(images.transpose(), 25)
reduced_images25 = reduced_images25

print("DONE: PCA.")



# (1b) Autoencoder
print("Autoencoder started.")

d = 4096

train_images = gray_images_dataframe.iloc[:, :]
test_images = gray_images_dataframe.iloc[:, :]

autoencoder100 = MLPRegressor(hidden_layer_sizes=(d, d//4, 100, d//4, d),
                           activation='tanh',
                           solver='adam',
                           learning_rate_init=0.0001,
                           max_iter=5,
                           tol=0.0000001,
                           verbose=True)

autoencoder100.fit(train_images, test_images)
reduced_autoencoder100 = autoencoder100.predict(train_images)
reduced_autoencoder100 = np.array(reduced_autoencoder100).transpose()


autoencoder50 = MLPRegressor(hidden_layer_sizes=(d, d//4, 50, d//4, d),
                           activation='tanh',
                           solver='adam',
                           learning_rate_init=0.0001,
                           max_iter=5,
                           tol=0.0000001,
                           verbose=True)

autoencoder50.fit(train_images, test_images)
reduced_autoencoder50 = autoencoder50.predict(train_images)
reduced_autoencoder50 = np.array(reduced_autoencoder50).transpose()

autoencoder25 = MLPRegressor(hidden_layer_sizes=(d, d//4, 25, d//4, d),
                           activation='tanh',
                           solver='adam',
                           learning_rate_init=0.0001,
                           max_iter=5,
                           tol=0.0000001,
                           verbose=True)

autoencoder25.fit(train_images, test_images)
reduced_autoencoder25 = autoencoder25.predict(train_images)
reduced_autoencoder25 = np.array(reduced_autoencoder25).transpose()

print("DONE: Autoencoder.")


# (2a) K-Means algorithms
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# K-Means Euclidean Distance

print("\n K-Means Euclidean Distance")

Kmeans = KMeans(init="random", n_clusters=10, n_init=10, max_iter=300, random_state=42)
predicted_labels_PCA100 = Kmeans.fit_predict(reduced_images100)
predicted_labels_PCA50 = Kmeans.fit_predict(reduced_images50)
predicted_labels_PCA25 = Kmeans.fit_predict(reduced_images25)
predicted_labels_Autoencoder50 = Kmeans.fit_predict(reduced_autoencoder50)
predicted_labels_Autoencoder25 = Kmeans.fit_predict(reduced_autoencoder25)
predicted_labels_Autoencoder100 = Kmeans.fit_predict(reduced_autoencoder100)


conf_matrix100 = confusion_matrix(real_labels, predicted_labels_PCA100)
conf_matrix50 = confusion_matrix(real_labels, predicted_labels_PCA50)
conf_matrix25 = confusion_matrix(real_labels, predicted_labels_PCA25)

sum100 = 0
max100 = 0
sum50 = 0
max50 = 0
sum25 = 0
max25 = 0


normalize_cm100 = np.round(conf_matrix100 / np.sum(conf_matrix100, axis=1).reshape(-1, 1), 4)
normalize_cm50 = np.round(conf_matrix50 / np.sum(conf_matrix50, axis=1).reshape(-1, 1), 4)
normalize_cm25 = np.round(conf_matrix25 / np.sum(conf_matrix25, axis=1).reshape(-1, 1), 4)

for i in range(10):

    sum100 = sum100 + np.sum(normalize_cm100[i])
    max100 = max100 + np.max(normalize_cm100[i])
    sum50 = sum50 + np.sum(normalize_cm50[i])
    max50 = max50 + np.max(normalize_cm50[i])
    sum25 = sum25 + np.sum(normalize_cm25[i])
    max25 = max25 + np.max(normalize_cm25[i])

purityscore100 = max100 / sum100
print("Purity Score of PCA M=100: ", purityscore100)
purityscore50 = max50 / sum50
print("Purity Score of PCA M=50: ", purityscore50)
purityscore25 = max25 / sum25
print("Purity Score of PCA M=25: ", purityscore25)

Fscore100 = f1_score(real_labels, predicted_labels_PCA100, average='weighted')
Fscore50 = f1_score(real_labels, predicted_labels_PCA50, average='weighted')
Fscore25 = f1_score(real_labels, predicted_labels_PCA25, average='weighted')

print("F-measure Score of PCA M=100: ", Fscore100)
print("F-measure Score of PCA M=50: ", Fscore50)
print("F-measure Score of PCA M=25: ", Fscore25)


print("Purity Score of Autoencoder M=100: ", purity_score(real_labels, predicted_labels_Autoencoder100))
print("F-measure Score of Autoencoder M=100: ", f1_score(real_labels, predicted_labels_Autoencoder100, average='weighted'))
print("Purity Score of Autoencoder M=50: ", purity_score(real_labels, predicted_labels_Autoencoder50))
print("F-measure Score of Autoencoder M=50: ", f1_score(real_labels, predicted_labels_Autoencoder50, average='weighted'))
print("Purity Score of Autoencoder M=25: ", purity_score(real_labels, predicted_labels_Autoencoder25))
print("F-measure Score of Autoencoder M=25: ", f1_score(real_labels, predicted_labels_Autoencoder25, average='weighted'))


# K-Means Cosine Distance
print("\n K-Means Cosine Distance")

cluster_num = 10
initial_centers100 = kmeans_plusplus_initializer(reduced_images100, cluster_num).initialize()
initial_centers50 = kmeans_plusplus_initializer(reduced_images50, cluster_num).initialize()
initial_centers25 = kmeans_plusplus_initializer(reduced_images25, cluster_num).initialize()
initial_centersAutoencoder100 = kmeans_plusplus_initializer(reduced_autoencoder100, cluster_num).initialize()
initial_centersAutoencoder50 = kmeans_plusplus_initializer(reduced_autoencoder50, cluster_num).initialize()
initial_centersAutoencoder25 = kmeans_plusplus_initializer(reduced_autoencoder25, cluster_num).initialize()

def cosine_distance(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    similiarity = np.dot(a, b.T) / (a_norm * b_norm)
    dist = 1. - similiarity
    return dist

# PCA Autoencoder M=100
metric100 = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
kmeans_instance100 = kmeans(reduced_images100, initial_centers100, metric=metric100)
kmeans_instance100.process()
clusters100 = kmeans_instance100.get_clusters()
cs100 = kmeans_instance100.get_centers()
predicted_labels_PCA100 = kmeans_instance100.predict(reduced_images100)

metricAutoencoder100 = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
kmeans_instanceAutoencoder100 = kmeans(reduced_autoencoder100, initial_centersAutoencoder100, metric=metricAutoencoder100)
kmeans_instanceAutoencoder100.process()
clustersAutoencoder100 = kmeans_instanceAutoencoder100.get_clusters()
csAutoencoder100 = kmeans_instanceAutoencoder100.get_centers()
predicted_labels_Autoencoder100 = kmeans_instanceAutoencoder100.predict(reduced_autoencoder100)

# PCA Autoencoder M=50
metric50 = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
kmeans_instance50 = kmeans(reduced_images50, initial_centers50, metric=metric50)
kmeans_instance50.process()
clusters50 = kmeans_instance50.get_clusters()
cs50 = kmeans_instance50.get_centers()
predicted_labels_PCA50 = kmeans_instance50.predict(reduced_images50)

metricAutoencoder50 = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
kmeans_instanceAutoencoder50 = kmeans(reduced_autoencoder50, initial_centersAutoencoder50, metric=metricAutoencoder50)
kmeans_instanceAutoencoder50.process()
clustersAutoencoder50 = kmeans_instanceAutoencoder50.get_clusters()
csAutoencoder50 = kmeans_instanceAutoencoder50.get_centers()
predicted_labels_Autoencoder50 = kmeans_instanceAutoencoder50.predict(reduced_autoencoder50)

# PCA Autoencoder M=25
metric25 = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
kmeans_instance25 = kmeans(reduced_images25, initial_centers25, metric=metric25)
kmeans_instance25.process()
clusters25 = kmeans_instance25.get_clusters()
cs25 = kmeans_instance25.get_centers()
predicted_labels_PCA25 = kmeans_instance25.predict(reduced_images25)

metricAutoencoder25 = distance_metric(type_metric.USER_DEFINED, func=cosine_distance)
kmeans_instanceAutoencoder25 = kmeans(reduced_autoencoder25, initial_centersAutoencoder25, metric=metricAutoencoder25)
kmeans_instanceAutoencoder25.process()
clustersAutoencoder25 = kmeans_instanceAutoencoder25.get_clusters()
csAutoencoder25 = kmeans_instanceAutoencoder25.get_centers()
predicted_labels_Autoencoder25 = kmeans_instanceAutoencoder25.predict(reduced_autoencoder25)


conf_matrix100 = confusion_matrix(real_labels, predicted_labels_PCA100)
conf_matrix50 = confusion_matrix(real_labels, predicted_labels_PCA50)
conf_matrix25 = confusion_matrix(real_labels, predicted_labels_PCA25)

normalize_cm100 = np.round(conf_matrix100 / np.sum(conf_matrix100, axis=1).reshape(-1, 1), 4)
normalize_cm50 = np.round(conf_matrix50 / np.sum(conf_matrix50, axis=1).reshape(-1, 1), 4)
normalize_cm25 = np.round(conf_matrix25 / np.sum(conf_matrix25, axis=1).reshape(-1, 1), 4)

for i in range(10):
    sum100 = sum100 + np.sum(normalize_cm100[i])
    max100 = max100 + np.max(normalize_cm100[i])
    sum50 = sum50 + np.sum(normalize_cm50[i])
    max50 = max50 + np.max(normalize_cm50[i])
    sum25 = sum25 + np.sum(normalize_cm25[i])
    max25 = max25 + np.max(normalize_cm25[i])


purityscore100 = max100 / sum100
print("Purity Score of PCA M=100: ", purityscore100)
purityscore50 = max50 / sum50
print("Purity Score of PCA M=50: ", purityscore50)
purityscore25 = max25 / sum25
print("Purity Score of PCA M=25: ", purityscore25)

Fscore100 = f1_score(real_labels, predicted_labels_PCA100, average='weighted')
Fscore50 = f1_score(real_labels, predicted_labels_PCA50, average='weighted')
Fscore25 = f1_score(real_labels, predicted_labels_PCA25, average='weighted')

print("F-measure Score of PCA M=100: ", Fscore100)
print("F-measure Score of PCA M=50: ", Fscore50)
print("F-measure Score of PCA M=25: ", Fscore25)


print("Purity Score of Autoencoder M=100: ", purity_score(real_labels, predicted_labels_Autoencoder100))
print("Purity Score of Autoencoder M=50: ", purity_score(real_labels, predicted_labels_Autoencoder50))
print("Purity Score of Autoencoder M=25: ", purity_score(real_labels, predicted_labels_Autoencoder25))
print("F-measure Score of Autoencoder M=100: ", f1_score(real_labels, predicted_labels_Autoencoder100, average='weighted'))
print("F-measure Score of Autoencoder M=50: ", f1_score(real_labels, predicted_labels_Autoencoder50, average='weighted'))
print("F-measure Score of Autoencoder M=25: ", f1_score(real_labels, predicted_labels_Autoencoder25, average='weighted'))


# (2b) Agglomerative Hierarchical Clustering
print("\n Agglomerative Hierarchical Clustering")
agglomerative_hierarchical_clustering = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
#predicted_labels_PCA100 = agglomerative_hierarchical_clustering.fit_predict(reduced_images100)
#predicted_labels_PCA50 = agglomerative_hierarchical_clustering.fit_predict(reduced_images50)
#predicted_labels_PCA25 = agglomerative_hierarchical_clustering.fit_predict(reduced_images25)
predicted_labels_Autoencoder100 = agglomerative_hierarchical_clustering.fit_predict(reduced_autoencoder100)
predicted_labels_Autoencoder50 = agglomerative_hierarchical_clustering.fit_predict(reduced_autoencoder50)
predicted_labels_Autoencoder25 = agglomerative_hierarchical_clustering.fit_predict(reduced_autoencoder25)


conf_matrix100 = confusion_matrix(real_labels, predicted_labels_PCA100)
conf_matrix50 = confusion_matrix(real_labels, predicted_labels_PCA50)
conf_matrix25 = confusion_matrix(real_labels, predicted_labels_PCA25)

sum100 = 0
max100 = 0
sum50 = 0
max50 = 0
sum25 = 0
max25 = 0


for i in range(10):

    sum100 = sum100 + np.sum(conf_matrix100[i])
    max100 = max100 + np.max(conf_matrix100[i])
    sum50 = sum50 + np.sum(conf_matrix50[i])
    max50 = max50 + np.max(conf_matrix50[i])
    sum25 = sum25 + np.sum(conf_matrix25[i])
    max25 = max25 + np.max(conf_matrix25[i])

purityscore100 = max100 / sum100
print("Purity Score of PCA M=100: ", purityscore100)
purityscore50 = max50 / sum50
print("Purity Score of PCA M=50: ", purityscore50)
purityscore25 = max25 / sum25
print("Purity Score of PCA M=25: ", purityscore25)

Fscore100 = f1_score(real_labels, predicted_labels_PCA100, average='weighted')
Fscore50 = f1_score(real_labels, predicted_labels_PCA50, average='weighted')
Fscore25 = f1_score(real_labels, predicted_labels_PCA25, average='weighted')

print("F-measure Score of PCA M=100: ", Fscore100)
print("F-measure Score of PCA M=50: ", Fscore50)
print("F-measure Score of PCA M=25: ", Fscore25)


print("Purity Score of Autoencoder M=100: ", purity_score(real_labels, predicted_labels_Autoencoder100))
print("Purity Score of Autoencoder M=50: ", purity_score(real_labels, predicted_labels_Autoencoder50))
print("Purity Score of Autoencoder M=25: ", purity_score(real_labels, predicted_labels_Autoencoder25))
print("F-measure Score of Autoencoder M=100: ", f1_score(real_labels, predicted_labels_Autoencoder100, average='weighted'))
print("F-measure Score of Autoencoder M=50: ", f1_score(real_labels, predicted_labels_Autoencoder50, average='weighted'))
print("F-measure Score of Autoencoder M=25: ", f1_score(real_labels, predicted_labels_Autoencoder25, average='weighted'))