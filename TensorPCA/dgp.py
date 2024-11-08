#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:53:29 2023

@author: junsupan
"""

import numpy as np
from numpy import linalg as LA
from scipy.stats import ortho_group
from .base import factor2tensor
from scipy.stats import random_correlation
import TensorPCA.auxiliary_functions as aux
import statsmodels.api as sm



def DGP(shape, R):
    """
    Generates the tensor data as a strong factor model introduced in the paper.
    The first dimension is treated as temporal, and generates AR(1) process factors.

    Parameters
    ----------
    shape : tupe
        shape of the tensor.
    R : int
        rank or number of factors.

    Returns
    -------
    Tensor.

    """
    d = len(shape)
    
    T = shape[0]
    
    # generating factor
    rho = 0.5
    sig_e = 0.1
    F = np.empty((T+100,R))
    e = sig_e * np.random.normal(0,1,(T+100,R))
    F[0,:] = e[0,:]
    for t in range(1,T+100):
        F[t,:] = F[t-1,:] * rho + e[t,:]
    F = F[100:,:]
    for r in range(R):
        F[:,r] = F[:,r] / LA.norm(F[:,r])
        
    # generating loadings
    M = [F]
    for j in range(1,d):
        M.append(ortho_group.rvs(shape[j])[:,0:R])
    print(f"np.shape(M) {np.shape(M[0])} second {np.shape(M[1])} third {np.shape(M[2])}")
    
    # generating scale component, singal strength
    s = np.sqrt(np.prod(shape)) * np.array(range(R,0,-1))
    # s = np.sqrt(np.prod(shape)) * np.array([2,0.8])
    
    Y = factor2tensor(s,M)
    
    # generating idiosyncratic noise
    sig_u = 0
    U = sig_u * np.random.normal(0,1,shape)
    
    # add noise to tensor Y
    Y = Y + U
    
    return Y, s, M

def DGP_alt(shape, R, ar_coefficients, seed, f_std = 1, error_std = 1):
    """
    Generates the tensor data as a strong factor model introduced in the paper.
    The first dimension is treated as temporal, and generates AR(1) process factors.

    Parameters
    ----------
    shape : tupe
        shape of the tensor.
    R : int
        rank or number of factors.

    Returns
    -------
    Tensor.

    """
    np.random.seed(seed)
    d = len(shape)
   
    T = shape[0]
   
    # generating factor
    f = np.zeros(T+100)
    eta = np.random.normal(0, f_std, T+100)
    if R > 1:
        f[:R-1] = np.random.normal(0, f_std, R-1)
        for t in range(R-1, T+100):
            f[t] = np.dot(ar_coefficients, f[t-R+1:t][::-1]) + eta[t]
    else:
        f = eta
    F = np.zeros((T,R))
    for r in range(R):
        if r != 0:
            F[:,r] = f[100-r:-r]/LA.norm(f[100-r:-r])
        else:
            F[:,r] = f[100:]/LA.norm(f[100:])
    #print(np.shape(F))
    # generating loadings
    M = [F]
    for j in range(1,d):
        M.append(ortho_group.rvs(shape[j])[:,0:R])
    # generating scale component, singal strength
    s = np.sqrt(np.prod(shape)) * np.array(range(R,0,-1))
    # s = np.sqrt(np.prod(shape)) * np.array([2,0.8])
    
    #   print(M)
    Y = factor2tensor(s,M)
  
    # generating idiosyncratic noise
    sig_u = 1
    U = sig_u * np.random.normal(0,error_std,shape)
  
    # add noise to tensor Y
    Y = Y + U
    return Y, s, M

def DGP_2d_dyn_fac(I,T, dynamic_factors, lags, ar_coefficients, noise_scale, seed):

    np.random.seed(seed)

    f = np.zeros((T,dynamic_factors))
    eta = np.random.normal(0, 0.1, (T,dynamic_factors))

    for factor in range(dynamic_factors):
        # Statsmodels requires us to specify the AR polynomial to generate an AR process, hence 
        # to get the AR coefficients we have to add a 1 to the start and then take the negative of our specified AR coefficients
        # Could rewrite later to accept just the ar_polynomial!!!
        ar_params = -1*ar_coefficients[:,factor]
        ar_params = np.concatenate(([1], ar_params) )
        ar_process = sm.tsa.arma_generate_sample(ar = ar_params, ma = [1],nsample = T, burnin = 0, scale = 1)  # AR(2), second value is thee lag polynomial for the MA bit of the process
        f[:,factor] = ar_process
    
    
    F = []
    for r in range(lags+1):
        f_slice = f[r:T-lags+r, :] if lags > 0 else f
        # If we normalise we lose the exact representation, so instead I just divide by the min of the two norms
        #f_slice = f_slice/min(np.linalg.norm(f_slice, axis = 0))
        for i in range(dynamic_factors):
            F.append(f_slice[:,i])
        
    F = np.array(F).T
    #F = F/np.linalg.norm(F,axis = 0)
    L = np.random.normal(0,1, size = (I, dynamic_factors*(lags+1)))
    #L = L/np.linalg.norm(L, axis = 0)
    M = [L,F]
    s = np.ones(shape = dynamic_factors*(lags+1))#[1,1]#np.sqrt(I*T) * np.random.normal(0,1, size = dynamic_factors*(lags+1))
    #Y = factor2tensor(s,M)
    Y = noise_scale * np.random.normal(0,1,(I,T-lags))
    for i in range(dynamic_factors*(lags+1)):
        Y += np.outer(L[:,i], F[:,i])
    s = np.linalg.norm(F,axis = 0) * np.linalg.norm(L, axis = 0)
    # generating idiosyncratic noise
    # add noise to tensor Y
    
    return Y, s, M

    

def gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda(I,J,T, seed, ar_coefficients, noise_scale, y_scale, dynamic_factors, lags,
    mu_mean, mu_sigma, lambda_mean, lambda_sigma, case = 1):
    """
    Generates a tensor based on specified parameters.
    With the difference that mu and Lambda can be correlated across the factors. E.g. With two dynamic factors and one lag the model is
    x_{ijt} = lambda_{i}^{0} mu_{j}^{0} f^{a}_{t} + lambda_{i}^{1} mu_{j}^{1} f^{a}_{t-1} +
              lambda_{i}^{2} mu_{j}^{2} f^{b}_{t} + lambda_{i}^{3} mu_{j}^{3} f^{b}_{t-1}
    We allow for lambda_{i}^{0} to be correlatd with lambda_{i}^{1} and lambda_{i}^{2} to be corrlated with lambda_{i}^{3} for each i separately.
    In the case with more lags we allow for the values for one dynamic factor to be correlated. Note that I assume no correlation between the i or j.
    Since correlation structures can be arbitrarily complex, I choose multivariate correlated normal. I provide two options, in the first (1)
    the correlations are the same for all i and all j, in the second (2) the correlations are random over the i and the j.
    In the second case, the means are generated randomly from a N(0,1) and the correlations are generated by first generating the eigenvalues
    of the correlation matrix as a size dynamic_factors*(lags+1) random normal over a uniform distribution on [0.1, 2] and then using scipy.stats.random_correlation
    to generate a random correlation matrix for each observation separately. 

    Args:
        I (int): Size of the first dimension.
        J (int): Size of the second dimension.
        T (int): Length of time series.
        seed (int): Random seed for reproducibility.
        ar_coefficients (array): Autoregressive coefficients for generating time series.
        noise_scale (float): Noise is a N(0,noise_scale^2) rdv
        y_scale (float): Each sigma_r is a N(0, y_scale^2) rdv (see paper for sigma_r)
        dynamic_factors (int): Number of dynamic factors in the time dimension.
        lags (int): Number of lags in the time factor. Note that if it is set to 0 then we have a purely static factor model
        mu_mean (array): Only valid if case == 1. Then the average of each Mu_{i}^{r} as explained above
        mu_sigma (2d array): only valid if case == 1. Then the correlation matrix for (Mu_{i}^{r})_{r in [|0,lags|] }
        lambda_mean (array): Only valid if case == 1. Then the average of each Lambda_{i}^{r} as explained above
        lambda_sigma (2d array): only valid if case == 1. Then the correlation matrix for (Lambda_{i}^{r})_{r in [|0,lags|] }
        case (int): Either 0 or 1. Correlation structure that Mu and Sigma are allowed to have. Either all datapoints are drawn from a 
                    multivariate normal with the above specificed mean and correlation matrix if case == 2. Or if case == 2 then for each i/j
                    the Mu and Lambda are drawn from a multivariate normal with a random correlation matrix and N(0,1) Mean

    Returns:
        tuple: Generated tensor Y, factor matrices L, M, and time series factors F.
    """
    if lags>0:
        assert len(ar_coefficients) == lags
    np.random.seed(seed)

    # Generate Lambda and Mu
    L = []
    M = []
    if case == 1:
        L = np.random.multivariate_normal(lambda_mean, lambda_sigma, size = I)
        M = np.random.multivariate_normal(mu_mean, mu_sigma, size = J)
    elif case == 2:
        for r in range(I):
            l_corr = random_correlation.rvs(eigs = aux.generate_uniform_sum_to_s(dynamic_factors*(lags+1)))
            l_mean = np.random.normal(0,1, size = dynamic_factors*(lags+1))
            L.append(np.random.multivariate_normal(l_mean, l_corr))
        for r in range(J):
            m_corr = random_correlation.rvs(eigs = aux.generate_uniform_sum_to_s(dynamic_factors*(lags+1)))
            m_mean = np.random.normal(0,1, size = dynamic_factors*(lags+1))       
            M.append(np.random.multivariate_normal(m_mean, m_corr))
        print(f"np.shape(M) {np.shape(M)} and type {type(M)}" )
    else:
        raise Exception("Make sure that the case is either 0 or 1, read the docstring for more information.")
    M = np.array(M)
    L = np.array(L)
    
    # Generate F
    f = np.zeros((T,dynamic_factors))
    eta = np.random.normal(0, 0.1, (T,dynamic_factors))
    
    """
    for factor in range(dynamic_factors):
        # Statsmodels requires us to specify the AR polynomial to generate an AR process, hence 
        # to get the AR coefficients we have to add a 1 to the start and then take the negative of our specified AR coefficients
        # Could rewrite later to accept just the ar_polynomial!!!
        ar_params = -1*ar_coefficients[:,factor]
        ar_params = np.concatenate(([1], ar_params) )
        ar_process = sm.tsa.arma_generate_sample(ar = ar_params, ma = [1],nsample = T, burnin = 0, scale = 0.001)  # AR(2), second value is thee lag polynomial for the MA bit of the process
        f[:,factor] = ar_process
    """
    f = np.zeros((T,dynamic_factors))
    eta = np.random.normal(0, 1, (T,dynamic_factors))

    if lags > 0:
        f[:lags,:] = np.random.normal(0, 1, (lags, dynamic_factors))
        for t in range(lags, T):
            #print(eta[t])
            #print(np.dot(ar_coefficients, f[t-R+1:t][::-1]))
            f[t,:] = np.sum(np.multiply(ar_coefficients, f[t-lags:t,:][::-1]), axis = 0) + eta[t, :]
    else:
        f = eta
    
    # Generate Y
    Y = noise_scale*np.random.normal(0, 1, (I, J, T-lags))
  
    F = []
    counter = 0
    s = []
    L_norm = np.linalg.norm(L, axis = 0)
    Mu_norm = np.linalg.norm(M, axis = 0)
    L = L/np.linalg.norm(L, axis = 0)    
    M = M/np.linalg.norm(M, axis = 0)
    
    for r in range(lags+1):
        f_slice = f[r:T-lags+r, :] if lags > 0 else f
        f_slice_norm = np.linalg.norm(f_slice, axis = 0)
        f_slice = f_slice/f_slice_norm
        for i in range(dynamic_factors):
            F.append(f_slice[:,i])
            
            #sigma = y_scale*np.random.normal(0,1)
            sigma = L_norm[counter]*Mu_norm[counter]*f_slice_norm[i]#np.linalg.norm(L_norm[counter])*np.linalg.norm(M[:,counter])*np.linalg.norm(f_slice[:,i])
            s.append(sigma)
            temp = sigma*np.multiply.outer(np.multiply.outer(L[:,counter], M[:,counter]),f_slice[:,i])
            #print(f"rank {counter} bit of Y {temp}")
            Y += temp#*sigma
            counter += 1

    F = np.array(F).T
    #F = F/np.linalg.norm(F, axis = 0)
    #print(f"Final Y first value {Y[0,0,0]}")
    return Y, [L, M, F], s
