from __future__ import division, print_function

import numpy as np
import pandas as pd
from ..utils import metrics
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy.linalg import matrix_rank
try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass


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
        

    def plot_fit1(self, size=(2,3), tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.S[n, :], 'b')
            if not axis_on:
                plt.axis('off')
    
    def plot_fit(self, tol=0.1, axis_on=True):

        n, d = self.D.shape

        

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

    
        plt.figure()

   
        plt.ylim((ymin - tol, ymax + tol))
        plt.plot(self.L[0, :] + self.S[0, :], 'r')
        plt.plot(self.S[0, :], 'b')
        if not axis_on:
            plt.axis('off')
                
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

    
    def predict(self, X):
        
        
        S = np.zeros(X.shape)
        L = self.L
        M = self.M
        # ,full_matrices=True
        U, s, V = np.linalg.svd(L)
           
       
        rank = matrix_rank(L)
#        U2 = U[:, rank:]
#        self.proj_C = np.dot(U2, U2.T)
        
#        U[np.absolute(U) < 0.0009] = 0
        U1  = U[:,:rank]
        I = np.identity(U1.shape[0], int)
        self.proj_C = np.dot(U1, U1.T)
        
#        V2 = V[rank:,:]
#        I = np.identity(V2.shape[0], int)
#        self.proj_C =I -  np.dot(V2, V2.T)
        
        

        
       
#        self.proj_C = I - np.dot(np.dot(U, np.linalg.inv(np.dot(U.T, U))) , U.T)
        L = np.dot(self.proj_C,X) 
#        for i in range(X.shape[0]):
 #              L[i,:]  = np.dot(self.proj_C, X[i,:]) 
        S = X - L
        y_pred = np.zeros(X.shape[0])
        
        w = np.zeros(S.shape[0])
        for i in range(S.shape[0]):
            w[i] =  np.std(S[i])
           
        for i in range(X.shape[0]):
           TF = np.dot(S[i],S[i]) >self.threshold
           if TF.any():
              y_pred[i] = 1
        self.y_pred = y_pred 
            
        return y_pred
    
    
    
    
    def calc_TP_FP_rate(self, y_true, y_pred):
    
    # Convert predictions to series with index matching y_true
        
     #   y_pred = pd.Series(y_pred, index=y_true.index)
        
        # Instantiate counters
        TP = 0
        FP = 0
        TN = 0
        FN = 0
    
        # Determine whether each prediction is TP, FP, TN, or FN
        for i in range(y_true.shape[0]): 
            if y_true[i]==y_pred[i]==1:
               TP += 1
            if y_pred[i]==1 and y_true[i]!=y_pred[i]:
               FP += 1
            if y_true[i]==y_pred[i]==0:
               TN += 1
            if y_pred[i]==0 and y_true[i]!=y_pred[i]:
               FN += 1
        
        tpr = TP / (TP + FN)
        if (FP + TN == 0):
            return tpr, 0.0
        # Calculate true positive rate and false positive rate
        
        fpr = FP / (FP + TN)

        return TP,FP,TN,FN

