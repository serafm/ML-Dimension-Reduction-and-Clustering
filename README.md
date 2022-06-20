# DimensionReductionAndClustering
On this link: https://www.kaggle.com/c/11785-spring2021-hw2p2s1-face-classification there is an experimental data set consisting of real images of persons.
In total there are 64x64 color (RGB) images of 4,000 faces, for each face there are several copies of the same face (with different point of view, brightness, etc.).
In this project are used randomly 10 of the existing persons (K = 10 categories) and for each person (category) are used 50 images(most of them faces have more than 50images).
So in total is a set of 10 x 50 = 500 images from K = 10 different persons. As the images are in color there are 3 channels of brightness (Red, Green, Blue)
and therefore for each pixel of the image there are 3 values (3 brightness levels with values between 0-255). So each image is described with 3 vectors (one for each channel)
of dimension 4,096 (64x64) each. A technique of converting a color image to monochrome without loss of information is by using the following linear relation:
0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue

### Dimension Reduction
- Reduce the dimension of the data using the PCA algorithm. For its dimension projection space test the values M = 100, 50, 25.
- Reduce the dimension of the data by training an Autoencoder with architecture (d - d/4 - M - d/4 - d), considering the same values of M.

### Face Clustering
Group the examples of the set into K=10 groups (considering one group per person) according to the following methods:
- k-means algorithm using either Euclidean distance or cosine distance.
- Agglomerative Hierarchical Clustering using the ward strategy for finding the teams with the shortest distance and joining them.

### Evaluation
The following two measures to evaluate grouping:
- Purity score
- F-measure
