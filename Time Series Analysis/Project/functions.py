import scipy
import numpy as np 
from sklearn.metrics.pairwise import pairwise_distances
from utility import calculate_l21_norm, build_similarity_graph, build_similarity_graph_classification, midd

#######################################################################################

def REFS(X, Y, alpha, max_iter, eps):
    """
    This function implements the REFS algorithm 
    It solves the problem:
    min_W||X W - Y||_2,1 + alpha||W||_2,1
    
    -------
    Reference: 
    https://papers.nips.cc/paper/2010/file/09c6c3783b4a70054da74f2538ed47c6-Paper.pdf
    
    """
    
    # Initialize variables 
    n_samples, n_features = X.shape
    objective_fct0 = 0
    
    # Construct A
    A = np.c_[X,  alpha*np.eye(n_samples)]
    
    # Initialize D0
    D0 = np.eye(n_features+n_samples)
    
    # Perform iterations
    for t in range(max_iter):
        
        temp = np.linalg.inv(D0) @ A.T
        U = temp @ np.linalg.inv(A @ temp) @ Y
        
        temp_ = np.sqrt(np.multiply(U, U).sum(axis = 1))
        
        # Delete small values 
        temp_[temp_ < 1e-16] = 1e-16
        
        D0 = np.diag(1/(2*temp_))
        
        # Define weights matrix 
        W = U[0:n_features, :]
        
        objective_fct1 = calculate_l21_norm(X @ W - Y) + alpha*calculate_l21_norm(W)

        if (t >= 1) and ((objective_fct1 - objective_fct0) < eps) :
            break
            
        objective_fct0  = objective_fct1

    return W

# --------------------------------------------------------------------- #

def UDFS(X, gamma, c, k, max_iter, eps):
    """
    This function implements the UDFS algorithm 
    It solves the problem:
    min_W Tr(W^T M W) + gamma ||W||_{2,1}, s.t. W^T W = I
    
    -------
    Reference: 
    https://www.ijcai.org/Proceedings/11/Papers/267.pdf
    
    """
    
    # Initialize variables 
    n_sample, n_feature = X.shape
    objective_fct0 = 0
    
    # Construct M
    I = np.eye(k+1)
    H = I - 1/(k+1) * np.ones((k+1, k+1))
    Mi = np.zeros((n_sample, n_sample))    
    
    for i in range(n_sample): 
        idx = np.argsort(pairwise_distances(X)[i])
        Xi = X.T[:, idx[0:k+1]]
        Xi_tilde = Xi @ H
        Bi = np.linalg.inv(Xi_tilde.T @ Xi_tilde + gamma*I)
        Si = np.zeros((n_sample, k+1))
        for q in range(k+1):
            Si[idx[q], q] = 1
        Mi += (Si @ H) @ (Bi @ H) @ Si.T
    
    M = X.T @ Mi @ X

    # Initialize D0
    D0 = np.eye(n_feature)

    # Perform iterations
    for t in range(max_iter):
        
        P = M + gamma*D0
        eigen_value, eigen_vector = scipy.linalg.eig(P)
        idx_ = np.argsort(eigen_value)[:c]
        W = eigen_vector[:, idx_]
    
        temp = np.sqrt(np.multiply(W,W).sum(axis = 1))
        temp[temp < 1e-16] = 1e-16
        D0 = np.diag(1/(2*temp))
        
        objective_fct1 = np.trace(W.T @ M @ W) + gamma*calculate_l21_norm(W)
        
        if (t >= 1) and ((objective_fct1 - objective_fct0) < eps) :
            break
        
        objective_fct0 = objective_fct1
   
    return W 

# --------------------------------------------------------------------- #

def laplacian_score(X, var=1.0, k =15):
    """
    This function implements the laplacian score feature selection

    -------
    Reference: 
    https://proceedings.neurips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf
    
    """

    # Initialize variables 
    n_sample, n_feature = X.shape
    I = np.ones((n_sample,1))
    Lr = np.zeros(n_feature)

    W = build_similarity_graph(X, var=var, k=k)
    D = np.diag(np.sum(W , axis=1))
    L = D - W
    
    for i in range(n_feature):

      feature = X[:,i].reshape(-1,1)
      temp = (feature.T @ D @ I)/(I.T @ D @ I) 

      # Compute fr tilde
      feature_tilde = feature - temp * I 

      # Compute Laplacian Score 
      Lr[i] = (feature_tilde.T @ L @ feature_tilde)/(feature_tilde.T @ D @ feature_tilde)

    return Lr

# --------------------------------------------------------------------- #

def fisher_score(X, Y) :
  """
  This function implements the fisher score feature selection

  -------
  Reference: 
  https://proceedings.neurips.cc/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf

  """

  # Initialize variables 
  n_sample, n_feature = X.shape
  I = np.ones((n_sample,1))
  Lr = np.zeros(n_feature)

  W = build_similarity_graph_classification(X,Y)
  D = np.diag(np.sum(W , axis=1))
  L = D - W 

  for i in range(n_feature):

      feature = X[:,i].reshape(-1,1)
      temp = (feature.T @ D @ I)/(I.T @ D @ I) 

      # Compute fr tilde
      feature_tilde = feature - temp * I 

      # Compute Laplacian Score 
      Lr[i] = (feature_tilde.T @ L @ feature_tilde)/(feature_tilde.T @ D @ feature_tilde)

  # Compute fisher score 
  fs = 1/Lr - 1

  return fs

# --------------------------------------------------------------------- #

def MIM(X,Y): 
  """
  This function implements the MIM feature selection

  -------
  Reference: 
  https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

  """

  # Initialize variables 
  n_sample, n_feature = X.shape
  mi = np.zeros(n_feature)

  # Compute Discrete Mutual Information vector 
  for i in range(n_feature):
    fi = X[:, i]
    mi[i] = midd(fi, Y)

  return mi 

# --------------------------------------------------------------------- #

def MIFS(X, Y, beta = 0.5): 
    """
    This function implements the MIFS feature selection

    -------
    Reference:
    https://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf

    """

    # Initialize variables 
    n_sample, n_feature = X.shape
    mi = np.zeros(n_feature)
    fmi = np.zeros(n_feature)
    slected_features_idx = []
    slected_features_score = []

    # Compute Discrete Mutual Information vector 
    mi = MIM(X,Y)

    # Select features 
    while len(slected_features_idx) < n_feature :
      if len(slected_features_idx) == 0 : 
        idx = np.argmax(mi)
        slected_features_idx.append(idx)
        slected_features_score.append(mi[idx])
        f_select = X[:, idx]

      value_max = -1e30
      idx_max = 0
      
      for i in range(n_feature): 
        if i not in slected_features_idx: 
          fmi[i] += midd(f_select, X[:,i]) 
          obj = mi[i] - beta*fmi[i]

          if obj > value_max : 
            value_max = obj 
            idx_max = i 

      slected_features_idx.append(idx_max)
      slected_features_score.append(value_max)
      f_select = X[:,idx_max]

    return slected_features_score , slected_features_idx

