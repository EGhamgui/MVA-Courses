import matplotlib.pyplot as plt
import scipy
import numpy as np
import networkx as nx
import random
import scipy.io
import scipy.spatial.distance as sd
from scipy.sparse.csgraph import minimum_spanning_tree


def min_span_tree(W):
    """
    Compute the minimum spanning tree of a graph.

    Parameters
    ----------
    W : array
        (n x n) adjacency or weight matrix representing the graph

    Returns
    -------
    T:  array
        (n x n) matrix such that T[i,j] = True if the edge (i, j) is in the min spanning tree, and
        T[i, j] = False otherwise
    """
    tree = minimum_spanning_tree(W).toarray()
    T = tree != 0
    return T


def build_similarity_graph(X, var=1, eps=0, k=0):
    """
    Computes the similarity matrix for a given dataset of samples.
     
    Parameters
    ----------
    X : array
        (n x m) matrix of m-dimensional samples
    var : double
        The sigma value for the exponential function, already squared
    eps : double
        Threshold eps for epsilon graphs
    k : int
        Number of neighbours k for k-nn. If zero, use epsilon-graph

    Returns
    -------
    W : array
        (n x n) dimensional matrix representing the weight matrix of the graph
    """

    n = X.shape[0]
    W = np.zeros((n, n))

    # euclidean distance squared between points
    dists = sd.squareform(sd.pdist(X, "sqeuclidean"))

    """
    Build full (similarity) graph. The similarity function is s(x,y)=exp(-||x-y||^2/var) 
        similarities: (n x n) matrix with similarities between all possible couples of points.
    """
    similarities = np.exp(-dists / var)

    # If epsilon graph
    if k == 0:
        """
        compute an epsilon graph from the similarities             
        for each node x_i, an epsilon graph has weights             
        w_ij = d(x_i,x_j) when w_ij >= eps, and 0 otherwise          
        """
        W = similarities
        W[W < eps] = 0

    # If kNN graph
    if k != 0:
        """
        compute a k-nn graph from the similarities                   
        for each node x_i, a k-nn graph has weights                  
        w_ij = d(x_i,x_j) for the k closest nodes to x_i, and 0     
        for all the k-n remaining nodes                              
        Remember to remove self similarity and                       
        make the graph undirected                                    
        """
        indices = np.argsort(-similarities, axis=1)[:, :k]
        values = np.take_along_axis(similarities, indices, axis=1)
        temp = np.zeros((n, n))
        np.put_along_axis(temp, indices, values, axis=1)  # asymmetric matrix
        W[temp != 0] = temp[temp != 0]
        W[temp.T != 0] = temp.T[temp.T != 0]

    return W


def build_laplacian(W, laplacian_normalization=""):
    """
    Compute graph Laplacian.

    Parameters
    ----------
    W : numpy array
        Adjacency matrix (n x n)
    laplacian_normalization : str
        String selecting which version of the laplacian matrix to construct.
            'unn':  unnormalized,
            'sym': symmetric normalization
            'rw':  random-walk normalization   

    Returns
    -------
    L: (n x n) dimensional matrix representing the Laplacian of the graph
    """

    degree = W.sum(1)
    if (not laplacian_normalization) or laplacian_normalization=='unn':
        return np.diag(degree) - W
    elif laplacian_normalization == "sym":
        aux = np.diag(1 / np.sqrt(degree))
        return np.eye(*W.shape) - aux.dot(W.dot(aux))
    elif laplacian_normalization == "rw":
        return np.eye(*W.shape) - np.diag(1 / degree).dot(W)
    else:
        raise ValueError


def build_laplacian_regularized(X, laplacian_regularization, var=1.0, eps=0.0, k=0, laplacian_normalization=""):
    """
    Function to construct a regularized Laplacian from data.

    Parameters
    ----------
    X : array
        (n x m) matrix of m-dimensional samples
    laplacian_regularization : double
        Regularization to add to the Laplacian (gamma)
    var : double
        The sigma value for the exponential (similarity) function, already squared.
    eps : double
        Threshold eps for epsilon graphs
    k : int
        Number of neighbours k for k-nn. If zero, use epsilon-graph.
    laplacian_normalization : str
        String selecting which version of the laplacian matrix to construct.
            'unn':  unnormalized,
            'sym': symmetric normalization
            'rw':  random-walk normalization   
    
    Returns
    -------
    Q : array
        (n x n ) matrix, the regularized Laplacian; Q = L + gamma*I,
        where gamma = laplacian_regularization.
    """
    # build the similarity graph W
    W = build_similarity_graph(X, var, eps, k)

    """
    Build the Laplacian L and the regularized Laplacian Q.
    Both are (n x n) matrices.
    """
    L = build_laplacian(W, laplacian_normalization)

    # compute Q
    Q = L + laplacian_regularization*np.eye(W.shape[0])

    return Q

def mask_labels(Y, l, per_class=False):
    """
    Function to select a subset of labels and mask the rest.

    Parameters
    ----------
    Y : array
        (n,) label vector, where entries Y_i take a value in [1, ..., C] , where C is the number of classes

    l : int
        Number of unmasked (revealed) labels to include in the output.
    
    per_class: bool, default: False
        If true, reveal l labels per class, instead of l labels in total.

    Returns
    -------
    Y_masked : array
        (n,) masked label vector, where entries Y_i take a value in [1, ..., C]
        if the node is labeled, or 0 if the node is unlabeled (masked)               
    """
    num_samples = np.size(Y, 0)

    """
     randomly sample l nodes to remain labeled, mask the others   
    """
    min_label = Y.min()
    max_label = Y.max()
    assert min_label == 1

    if not per_class:
        # reveal l labels in total
        Y_masked = np.zeros(num_samples)
        indices_to_reveal = np.arange(num_samples)
        np.random.shuffle(indices_to_reveal)
        indices_to_reveal = indices_to_reveal[:l]
        Y_masked[indices_to_reveal] = Y[indices_to_reveal]
    else:
        # reveal l labels per class
        Y_masked = np.zeros(num_samples)
        for label in range(min_label, max_label+1):
            indices = np.where( Y == label)[0]
            np.random.shuffle(indices)
            indices = indices[:l]
            Y_masked[indices] = Y[indices]

    return Y_masked

def plot_edges_and_points(X, Y, W, title='',
                         points_to_highlight_green=None,
                         points_to_highlight_yellow=None):
    colors=['go-','ro-','co-','ko-','yo-','mo-']
    colors_highlight_green=['gX']*len(colors)
    colors_highlight_yellow=['yX']*len(colors)


    n=len(X)
    G=nx.from_numpy_matrix(W)
    nx.draw_networkx_edges(G,X)
    for i in range(n):
        plt.plot(X[i,0],X[i,1],colors[int(Y[i])])
    
    if points_to_highlight_green is not None:
        for i in points_to_highlight_green:
            plt.plot(X[i,0],X[i,1],colors_highlight_green[int(Y[i])], markersize=17)

    if points_to_highlight_yellow is not None:
        for i in points_to_highlight_yellow:
            plt.plot(X[i,0],X[i,1],colors_highlight_yellow[int(Y[i])], markersize=17)
    plt.title(title)
    plt.axis('equal')

            
def plot_graph_matrix(X, Y, W):
    plt.figure()
    plt.clf()
    plt.subplot(1, 2, 1)
    plot_edges_and_points(X,Y,W)
    plt.subplot(1, 2, 2)
    plt.imshow(W, extent=[0, 1, 0, 1])
    plt.show()           

            
def plot_classification(X, Y, Y_masked, noise_indices, labels, var=1, eps=0, k=0):
    plt.figure(figsize=(12, 5))
    W = build_similarity_graph(X, var=var, eps=eps, k=k)

    revealed = np.where( Y_masked != 0)[0]

    plt.subplot(1, 3, 1)
    plot_edges_and_points(X, Y, W, 'ground truth')

    plt.subplot(1, 3, 2)
    plot_edges_and_points(X, labels, W, 'HFS')
    plt.subplot(1, 3, 3)
    plot_edges_and_points(X, labels, W, 'HFS - Revealed Labels',
                          points_to_highlight_green=revealed,
                          points_to_highlight_yellow=noise_indices)
    plt.show()         

    
def label_noise(Y, alpha):
    ind = np.arange(len(Y))
    random.shuffle(ind)
    Y[ind[:alpha]] = 3-Y[ind[:alpha]]
    return Y

if __name__ == '__main__':
    Y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    print(mask_labels(Y, 2))
    print(mask_labels(Y, 2, per_class=True))
