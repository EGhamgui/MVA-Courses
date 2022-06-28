import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skm
import scipy

from utils import plot_clustering_result, plot_the_bend
from build_similarity_graph import build_similarity_graph
from generate_data import blobs, two_moons, point_and_circle


def build_laplacian(W, laplacian_normalization=""):
    """
    Compute graph Laplacian.

    :param W: adjacency matrix
    :param laplacian_normalization:  string selecting which version of the laplacian matrix to construct
                                     'unn':  unnormalized,
                                     'sym': symmetric normalization
                                     'rw':  random-walk normalization
    :return: L: (n x n) dimensional matrix representing the Laplacian of the graph
    """
    return np.zeros(W.shape)


def spectral_clustering(L, chosen_eig_indices, num_classes=2):
    """
    :param L: Graph Laplacian (standard or normalized)
    :param chosen_eig_indices: indices of eigenvectors to use for clustering
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.sparse.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    E = None
    U = None

    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]                   
    """
    Y = None
    return Y


def two_blobs_clustering():
    """
    TO BE COMPLETED

    Clustering of two blobs. Used in questions 2.1 and 2.2
    """

    # Get data and compute number of classes
    X, Y = blobs(600, n_blobs=2, blob_var=0.15, surplus=0)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 3
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    chosen_eig_indices = [1, 2, 3]    # indices of the ordered eigenvalues to pick

    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)

    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    # Plot results
    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def choose_eigenvalues(eigenvalues):
    """
    Function to choose the indices of which eigenvalues to use for clustering.

    :param eigenvalues: sorted eigenvalues (in ascending order)
    :return: indices of the eigenvalues to use
    """
    eig_ind = [1, 2, 3]
    return eig_ind


def spectral_clustering_adaptive(L, num_classes=2):
    """
    Spectral clustering that adaptively chooses which eigenvalues to use.
    :param L: Graph Laplacian (standard or normalized)
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    E = None
    U = None

    """
    compute the clustering assignment from the eigenvectors   
    Y = (n x 1) cluster assignments [1,2,...,c]                   
    """
    Y = None
    return Y


def find_the_bend():
    """
    TO BE COMPLETED

    Used in question 2.3
    :return:
    """

    # the number of samples to generate
    num_samples = 600

    # Generate blobs and compute number of clusters
    X, Y = blobs(num_samples, 4, 0.2)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 2
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = ''  # either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization


    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)

    """
    compute first 15 eigenvalues and call choose_eigenvalues() to choose which ones to use. 
    """
    eigenvalues = np.zeros(15)
    chosen_eig_indices = choose_eigenvalues(eigenvalues)  # indices of the ordered eigenvalues to pick


    """
    compute spectral clustering solution using a non-adaptive method first, and an adaptive one after (see handout) 
    Y_rec = (n x 1) cluster assignments [0,1,..., c-1]    
    """
    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
    Y_rec_adaptive = spectral_clustering_adaptive(L, num_classes=num_classes)

    plot_the_bend(X, Y, L, Y_rec, eigenvalues)


def two_moons_clustering():
    """
    TO BE COMPLETED.

    Used in question 2.7
    """
    # Generate data and compute number of clusters
    X, Y = two_moons(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 3
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    chosen_eig_indices = [1, 2, 3]    # indices of the ordered eigenvalues to pick


    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def point_and_circle_clustering():
    """
    TO BE COMPLETED.

    Used in question 2.8
    """
    # Generate data and compute number of clusters
    X, Y = point_and_circle(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 3
    var = 1.0  # exponential_euclidean's sigma^2

    chosen_eig_indices = [1, 2, 3]    # indices of the ordered eigenvalues to pick


    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L_unn = build_laplacian(W, 'unn')
    L_norm = build_laplacian(W, 'sym')

    Y_unn = spectral_clustering(L_unn, chosen_eig_indices, num_classes=num_classes)
    Y_norm = spectral_clustering(L_norm, chosen_eig_indices, num_classes=num_classes)

    plot_clustering_result(X, Y, L_unn, Y_unn, Y_norm, 1)


def parameter_sensitivity():
    """
    TO BE COMPLETED.

    A function to test spectral clustering sensitivity to parameter choice.

    Used in question 2.9
    """
    # the number of samples to generate
    num_samples = 500

    """
    Choose parameters
    """
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'unn'
    chosen_eig_indices = [1, 2]

    """
    Choose candidate parameters
    """
    parameter_candidate = [1, 2]  # the number of neighbours for the graph or the epsilon threshold
    parameter_performance = []

    for k in parameter_candidate:
        # Generate data
        X, Y = two_moons(num_samples, 1, 0.02)
        num_classes = len(np.unique(Y))

        W = build_similarity_graph(X, k=k)
        L = build_laplacian(W, laplacian_normalization)

        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)

        parameter_performance += [skm.adjusted_rand_score(Y, Y_rec)]

    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity')
    plt.show()

