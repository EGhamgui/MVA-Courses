import numpy as np 
from scipy.stats import entropy

#######################################################################################

def calculate_l21_norm(X):
    """
    This function returns the L_2,1 norm of the signal X

    Parameters
    ----------
    X : numpy array
        (n x m) matrix of m-dimensional samples

    """
    return (np.sqrt(np.multiply(X, X).sum(axis = 1))).sum()

# --------------------------------------------------------------------- #

def build_similarity_graph(X, var=1.0, k=0):
    """
    This function computes the similarity matrix for a given dataset of samples
   
    Parameters
    ----------
    X : numpy array
        (n x m) matrix of m-dimensional samples
    var : double 
        the sigma value for the exponential function, already squared
    k : int
        The number of neighbours k for k-nn 

    Returns
    -------
    W: (n x n) dimensional matrix representing the weight matrix of the graph

    """
    n = X.shape[0]
    W = np.zeros((n, n))

    """
    Build similarity graph
    similarities: (n x n) matrix with similarities between all possible couples of points
    The similarity function is d(x,y)=exp(-||x-y||^2/(2*var))
    """
    similarities = np.zeros((n, n))
    for i in range(n): 
        for j in range(n): 
            if i!=j : 
                similarities[i,j] = np.exp(-np.linalg.norm(X[i]-X[j])**2/(2*var))
    """
    Compute a k-nn graph from the similarities                   
    for each node x_i, a k-nn graph has weights                  
    w_ij = d(x_i,x_j) for the k closest nodes to x_i, 
    and 0 for all the k-n remaining nodes                                                            
    """
    for i in range(n):
        k_nn_i = np.argsort( -similarities[i,:])[:k] 
        for j in k_nn_i: 
            W[i,j] = similarities[i,j]
            W[j,i] = similarities[i,j]
                   
    return W

# --------------------------------------------------------------------- #

def build_similarity_graph_classification(X, Y):
    """
    This function computes the similarity matrix for a given dataset of samples
   
    Parameters
    ----------
    X : numpy array
        (n x m) matrix of m-dimensional samples
    Y : numpy array
        (n) vector of classes 

    Returns
    -------
    W : (n x n) dimensional matrix representing the weight matrix of the graph

    """
    n = X.shape[0]
    W = np.zeros((n, n))

    for i in range(n): 
      for j in range(n):
        if (i!=j) and (Y[i] == Y[j]): 
          W[i,j] = 1/np.sum(Y==Y[i])

    return W

# --------------------------------------------------------------------- #

def hist(X):
    """
    Compute the histogram from list of samples

    Parameters
    ---------- 
    X : numpy array
        feature vector 

    """
    d = dict()
    for s in X:
        d[s] = d.get(s, 0) + 1

    return list(map(lambda z: float(z)/len(X), d.values()))

# --------------------------------------------------------------------- #

def entropy_value(X):
    """
    This function computes the entropy of a feature vector

    Parameters
    ---------- 
    X : numpy array
        feature vector 

    """
    histo = hist(X)
    return entropy(histo, base=2)

# --------------------------------------------------------------------- #

def midd(x, y):
    """
    This function computes Discrete Mutual Information estimator
    
    Parameters
    ---------- 
    x : numpy array
        first feature vector
    y : numpy array
        second feature vector

    """
    Hx = entropy_value(x)
    Hy = entropy_value(y)
    Hxy = entropy_value(list(zip(x, y)))
    
    return Hx - (Hxy - Hy)