from __future__ import division, print_function

import numpy as np
import pandas as pd
from ..utils import metrics
import matplotlib
from sklearn import preprocessing
from numpy.linalg import matrix_rank

class R_pca:

    def __init__(self, D, mu=None, lmbda=None, threshold=None, n_components= 0.95):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.M = np.zeros(self.D.shape)
        self.threshold = threshold
        self.var = []
        self.n_components = n_components
        
        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu
        
        if lmbda:
            self.lmbda = lmbda
        
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
            
            
  
    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        M_cov = np.dot(M.T, M) / float(M.shape[0])
        U, S, V = np.linalg.svd(M, full_matrices=False)
        self.M = M
        sigma = S
        total_variance = np.sum(sigma)
        n_components = self.n_components
        comp_list = []
        if n_components < 1:
            variance = 0
            for i in range(M.shape[1]):
                variance += sigma[i]
                self.var.append(variance / total_variance)
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1
            comp_list.append(n_components)
        self.M = M
    #    plt.plot(comp_list, self.var)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit2(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
       

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        
      
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)                           #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            
         #   if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
          #      print('iteration: {0}, error: {1}'.format(iter, err))

      #  U, S, V = np.linalg.svd(Lk, full_matrices=False)
      #  I = np.identity(self.num_events, int)
      #  self.proj_C = I - np.dot(U.T, U)

                
        self.L = Lk
        self.S = Sk
        return Lk, Sk
        

                
    def evaluate1(self, X, y_true):
           print('====== Evaluation summary ======')
           y_pred = np.zeros(X.shape[0])
           S = self.S
           L = self.L
         
           for i in range(X.shape[0]):
              
               TF = S[i] > self.threshold
               if TF.any():
                  y_pred[i] = 1
           precision, recall, f1, accuracy = metrics(y_pred, y_true)
           print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, Accuracy: {:.5f} \n'.format(precision, recall, f1, accuracy))
           return precision, recall, f1, accuracy, y_pred
            
                
    def evaluate2(self, X, y_true):
           print('====== Evaluation summary ======')
           y_pred = self.predict(X)
           precision, recall, f1, accuracy = metrics(y_pred, y_true)
           print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}, Accuracy: {:.5f} \n'.format(precision, recall, f1, accuracy))
           return precision, recall, f1, accuracy, y_pred

    

    
    
    
    
    
