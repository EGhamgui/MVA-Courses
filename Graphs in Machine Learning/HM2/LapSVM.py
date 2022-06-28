#!/usr/bin/env python
# -*- coding: utf-8 -*-



#=========================================================================================================
#================================ 0. MODULE


import numpy as np
import math
from numpy import linalg

import sklearn
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph

import scipy.optimize as sco

from itertools import cycle, islice


#=========================================================================================================
#================================ 1. ALGORITHM


class LapSVM(object):

    def __init__(self, n_neighbors, kernel, lambda_k, lambda_u):
        """
        Laplacian Support Vector Machines

        Parameters
        ----------
        n_neighbors : integer
            Number of neighbors to use when constructing the graph
        lambda_k : float
        lambda_u : float
        """
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
    

    def fit(self, X, X_no_label, Y):
        """
        Fit the model
        
        Parameters
        ----------
        X : ndarray shape (n_labeled_samples, n_features)
            Labeled data
        X_no_label : ndarray shape (n_unlabeled_samples, n_features)
            Unlabeled data
        Y : ndarray shape (n_labeled_samples,)
            Labels
        """
        # Storing parameters
        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u
        
        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        Y = np.diag(Y)
        
        # Memory optimization
        del X_no_label
        
        # Building adjacency matrix from the knn graph
        #print('Computing adjacent matrix', end='...')
        W = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity')
        W = (((W + W.T) > 0) * 1)
        #print('done')

        # Computing Graph Laplacian
        #print('Computing laplacian graph', end='...')
        L = np.diag(W.sum(axis=0)) - W
        #print('done')

        # Computing K with k(i,j) = kernel(i, j)
        #print('Computing kernel matrix', end='...')
        K = self.kernel(self.X)
        #print('done')

        # Creating matrix J [I (l x l), 0 (l x (l+u))]
        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)

        ###########################################################################
        
        # Computing "almost" alpha
        #print('Inverting matrix', end='...')
        almost_alpha = np.linalg.inv(2 * self.lambda_k * np.identity(l + u) \
                                     + ((2 * self.lambda_u) / (l + u) ** 2) * L.dot(K)).dot(J.T).dot(Y)
        
        # Computing Q
        Q = Y.dot(J).dot(K).dot(almost_alpha)
        #print('done')
        
        # Memory optimization
        del W, L, K, J
        
        # Solving beta using scypy optimize function
        
        #print('Solving beta', end='...')
        
        e = np.ones(l)
        q = -e
        
        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)
        
        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))
        
        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]
        
        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))
        
        def constraint_grad(beta):
            return np.diag(Y)
        
        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}
        
        # ===== Solving =====
        x0 = np.zeros(l)
        
        beta_hat = sco.minimize(objective_func, x0, jac=objective_grad, \
                                constraints=cons, bounds=bounds, method='L-BFGS-B')['x']
        #print('done')
        
        # Computing final alpha
        #print('Computing alpha', end='...')
        self.alpha = almost_alpha.dot(beta_hat)
        #print('done')
        
        del almost_alpha, Q
        
        ###########################################################################
        
        # Finding optimal decision boundary b using labeled data
        new_K = self.kernel(self.X, X)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == np.diag(Y)) / len(predictions))
        
        bs = np.linspace(0, 1, num=101)
        res = np.array([to_minimize(b) for b in bs])
        self.b = bs[res == np.min(res)][0]
    

    def predict(self, Xtest):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        predictions : ndarray shape (n_samples, )
            Predicted labels for Xtest
        """

        # Computing K_new for X
        new_K = self.kernel(self.X, Xtest)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        predictions = np.array((f > self.b) * 1)
        return predictions
    

    def accuracy(self, Xtest, Ytrue):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
        Ytrue : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(Xtest)
        accuracy = sum(predictions == Ytrue) / len(predictions)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))
