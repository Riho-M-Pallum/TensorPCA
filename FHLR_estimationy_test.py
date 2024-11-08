import numpy as np
import pandas as pd
import datetime as dt
import DynamicFactorAnalysis.dynamicfactoranalysis as dfa
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from TensorPCA.dgp import DGP_2d_dyn_fac
import TensorPCA.auxiliary_functions as aux

a = np.linspace(0,10, 11)
print(a)
print(a[:0:-1])


seed = 1# np.random.randint(0,10000)# 990 # 609, 36, 990, 8550,2221# 
print(f"seed = {seed}")
Lags = 1
ar_coeffs = np.array([[0.8, 0.5]])
# Ok so, for recostruction the scale components are always positive. While the true rotation is sometimes one positive ad none negative
# e.g. true is [1,-1], I recover [1,1]
n_components = 2
I = 10
T = 15 + Lags 

Y, s, M = DGP_2d_dyn_fac(I,T, n_components, Lags, ar_coeffs, noise_scale = 0, seed = seed)
F = M[1]
Lambda = M[0]

print(np.shape(Y))



M, cov, idio = aux.FHLR_2D_estimation(Y, 2, 1, 10, 4)
F_hat = M[1]
Lambda_hat = M[0]

print("Here comes the sun")

#print(cont_eigenvecs[:,0] @ cont_idio @ cont_eigenvecs[:,0].T)
print(f"The shape of Y {np.shape(Y)}")
print(f"This is the F_hat \n{F_hat},\n this is the true F\n {F}")
print(f"This is the shape of Fhat {np.shape(F_hat)}")
print(f"F rank is {np.linalg.matrix_rank(F_hat)}")
print(f"Now to see if I did it correctly by taking the cos error {aux.get_cos_errors(F, F_hat)}")
