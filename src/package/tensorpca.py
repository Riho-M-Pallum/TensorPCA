#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:32:40 2023

@author: junsupan
@editor: Riho Marten Pallum
Changes: 
    1) Moved all instance attributes so they declared inside the __init__
    2) Removed self.unfolded attrbiute and made an unfolding be accessible through a getter method unfold to avoid storing
        duplicate data. The idea is that we are also storing the tensor itself. The only time that we look at the unfolding is
        when we run the tpca algorithm itself. Therefore, we can just generate the unfoldings in the tpca algorithm and not
        worry about storing them. 
    3) Made all dictionaries that take a number as a key into lists
    4) Made implicit casts to numpy for ndarray attributes implicit
"""

import numpy as np
from numpy import linalg as LA
from package import auxiliary_functions as aux


        

class TensorPCA:
    
    def __init__(self, tensor):
        """
        Sets up initial parameters

        Parameters
        ----------
        tensor : array_like
            Tensor data.

        Returns
        -------
        None.

        """
        
        if np.isnan(tensor).any() == True:
            raise ValueError('The tensor contains missing values')
            
        self.tensor = tensor # store tensor itself
        self.order = np.ndim(tensor) # order of the tensor, implicit numpy
        self.shape = np.shape(tensor) # shape of the tensor, implicit numpy
        
        self.s_hat = [0 for _ in range(self.order)]
        self.m_hat = [0 for _ in range(self.order)]
        self.rotated_m = [0 for _ in range(self.order)]
        

        self.S = np.empty(self.order)
        self.p = np.empty(self.order)
        
        
    
    def unfold(self, mode):
        """
        Returns unfolded tensor in jth mode

        Parameters
        ----------
        mode : int
            jth dimension of the tensor, j <= d.

        Returns
        -------
        array
            unfolded tensor.

        """

        return np.moveaxis(self.tensor,mode,0) \
            .reshape((self.shape[mode],-1),order='F')
    
    
    def t_pca(self, R):
        """
        Calculates the tensor pca components

        Parameters
        ----------
        R : int
            number of factors.

        Returns
        -------
        dict
            estimated scale components for each mode.
        dict
            estimated vector components for each mode.

        """

        
        # Estimates Tensor PCA
        for mode in range(self.order):
            s, gamma = LA.eigh(self.unfold(mode) @ self.unfold(mode).transpose()) # Eigen-decomposition
            self.s_hat[mode] = np.sqrt(np.sort(s)[::-1][:R]).real # scale components
            self.m_hat[mode] = gamma[:,s.argsort()[::-1][:R]].real # vector components
            
        return self.s_hat, self.m_hat
            
    
    def ranktest(self, TW_dist):
        """
        Hypothesis Testing:
            Null: rank <= k
            Alternative: k < rank <= K
        where rank means the number of factors
        
        

        Parameters
        ----------
        TW_dist : tuple contains k, K, and the approximated distribution
            approximated distribution of statistic, run "dist" function first.

        Returns
        -------
        array
            Test statistics in each mode (dimension).
        array
            p-values of the statistics in each mode.

        """
        k = TW_dist[0]
        K = TW_dist[1]
        M = len(TW_dist[2])
        dist = TW_dist[2]
            
        # Test for each dimension
        for mode in range(self.order):
            # Calculates eigen value for each dimension
            s, _ = LA.eigh(self.unfolded[str(mode)] @ self.unfolded[str(mode)].transpose())
            s = np.sort(s)[::-1]
            
            # Calculates the test statistic
            eig_ratio = np.empty(K-k)
            for r in range(K-k):
                eig_ratio[r] = (s[k+r] - s[k+r+1])/(s[k+r+1] - s[k+r+2])
            
            self.S[mode] = max(eig_ratio)
            self.p[mode] = sum(dist > self.S[mode])/M
            
        return self.S, self.p

    def ALS(self, convergence_criteria, max_iters = 1000):
        unrotated_data = [self.m_hat[i] for i in range(len(self.m_hat))]
        self.rotated_factors, counter = aux.alternating_least_squares(self.tensor, unrotated_data,
            convergence_criteria = convergence_criteria, max_iters = max_iters)
        print(f"ALS took {counter} iterations to converge")
        return self.rotated_factors