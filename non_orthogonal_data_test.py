#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# To do: Run simulations with correlated Mu and Lambda over time. See if I can recover AR process, simplify model and go to the 2D case and see whether
# We can replicate everything there (for simplicity of presenting results higher order cases can be covered later).

Example code

See 'simulation.py' for a simulation in Hypothesis testing

"""
import numpy as np
import pandas as pd
from TensorPCA.tensorpca import TensorPCA as tPCA
from TensorPCA.dgp import DGP
from TensorPCA.dgp import gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import TensorPCA.auxiliary_functions as aux
import os
from scipy.linalg import null_space
import plotly.express as px
from scipy.stats import random_correlation


seed = np.random.randint(0,10000)# 990 # 609, 36, 990, 8550,2221# 
print(f"seed = {seed}")
Lags = 3
ar_coeffs = np.array([[0.8, 0.5], [0.1,0.4], [0., 0.2]])
n_components = 2


resulting_errors = {"j":[]}
resulting_correlations = {"j":[]}
recovered_F = {"True": [], "Recovered":[], "i": [], "j": []}
I = 30
J = 40
T = 60 + Lags

# So if we increase the time dimension too much with respect to the others then we do not recover all the factors well
# Something weird is going on, sometimes we recover the last two factors well and sometimes we recover them really badly, seems to be seed dependent.
# That is most likely going to be small sample problems.

Mu_corr = random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Sigma_corr = random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Mu_m = np.random.normal(0,1, size = n_components*(Lags+1))
Lambda_m = np.random.normal(0,1, size = n_components*(Lags+1))

Y, M = gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda(I,J,T, seed, ar_coefficients = ar_coeffs, noise_scale = 1, y_scale = 5,
        dynamic_factors = n_components, lags = Lags, mu_mean = Mu_m, mu_sigma = Mu_corr, lambda_mean = Lambda_m, lambda_sigma = Sigma_corr, case = 1)
# Note that the data I generate is very sensitive to noise as I normalise all the factors. So what we have is basically some number that
# is between 0 and 1 to the power of three, which makes the factors very small. So even if we multiply by e.g. 5 it's still very small 
# To avoid this problem in the original DGP function they multiply the signal strength by the total number of dimensions i.e. I*J*T


Z = tPCA(Y)
s_hat, M_hat = Z.t_pca(n_components*(Lags+1))
# True factors
Lambda = M[0]
Mu = M[1]
F = M[2]
print(f"We have Lambda has shape {np.shape(Lambda)}, Mu has shape {np.shape(Mu)} and F has shape {np.shape(F)}")

#Estimated factors
Lambda_hat = M_hat["0"]
Mu_hat = M_hat["1"]
F_hat = M_hat["2"]

print("F R2 error")
print(aux.get_R2_errors(F, F_hat))
print("F cos error")
print(aux.get_cos_errors(F, F_hat))
print("F L21 error")
print(aux.get_L21_errors(F, F_hat))

print("Lambda R2 error")
print(aux.get_R2_errors(Lambda, Lambda_hat))
print("Lambda cos error")
print(aux.get_cos_errors(Lambda, Lambda_hat))

print("Mu R2 error")
print(aux.get_R2_errors(Mu, Mu_hat))
print("Mu cos error")
print(aux.get_cos_errors(Mu, Mu_hat))


print("Now to recover a rotaton")
recovered = aux.recover_lags(F_hat, lags = Lags, dynamic_factors = n_components)

print(f"recovered rotation R2 error \n{aux.get_R2_errors(F, recovered)}")
print(f"recovered rotation cos error \n{aux.get_cos_errors(F, recovered)}")
print("normalised recovered L21 error")
print(aux.get_L21_errors(F, recovered/np.linalg.norm(recovered, axis = 0)))

print(f"Whether the recovered rotation falls into the original recovered \n{aux.get_cos_errors(F_hat, recovered)}")
# Do we recover the contemporaneous and the lags well
#print(f"first recovred factor \n{recovered[:,[0,1,2,3]]}\nSecond recovered factor \n{recovered[:,[4,5,6,7]]}")
cont_vars = F[:,[0,1]]
lag_vars = F[:,[2,3]]
second_lag_vars = F[:,[4,5]]
third_lag_vars = F[:,[6,7]]
rec_cont_vars = recovered[:,[3,7]]
rec_lag_vars = recovered[:,[2,6]]
rec_two_lag_vars = recovered[:,[1,5]]
rec_three_lag_vars = recovered[:,[0,4]]
print(f"Projection of recovered contemporaneous to true contemporaneous \n{aux.get_cos_errors(cont_vars, rec_cont_vars)}")
print(f"Projection of recovered one lagged to true lagged \n{aux.get_cos_errors(lag_vars, rec_lag_vars)}")
print(f"Projection of recovered two lagged to true lagged \n{aux.get_cos_errors(second_lag_vars, rec_two_lag_vars)}")
print(f"Projection of recovered three lagged to true lagged \n{aux.get_cos_errors(third_lag_vars, rec_three_lag_vars)}")

print(f"Compared to just F_hat on contemporaneous \n{aux.get_cos_errors(cont_vars, F_hat)}")
print(f"Compared to just F_hat on lagged \n{aux.get_cos_errors(lag_vars, F_hat)}")
