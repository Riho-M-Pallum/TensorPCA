import numpy as np
import pandas as pd
from TensorPCA.tensorpca import TensorPCA as tPCA
from TensorPCA.dgp import DGP
from TensorPCA.dgp import DGP_2d_dyn_fac
from TensorPCA.dgp import gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import TensorPCA.auxiliary_functions as aux
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import random_correlation


def fit_var(F):
	model = sm.VAR(F)
	try:
		lag_order = model.select_order(maxlags = 5)
	except:
		lag_order = 1 
		print(f"Model lag order {lag_order}")
		fitted_model = model.fit(lag_order)	
		return fitted_model, lag_order
	print(f"Model lag order {lag_order}")
	fitted_model = model.fit(lag_order.aic)	
	return fitted_model, lag_order.aic


def tensorPCA_predict(Y, steps, components):
	Z = tPCA(Y)
	s_hat, M_hat = Z.t_pca(components)
	Lambda_hat = M_hat["0"]
	Mu_hat = M_hat["1"]
	F_hat = M_hat["2"]
	print(f"tensorPCA F covariance {F_hat.T @ F_hat}")
	F_var_model, lag_order = fit_var(F_hat)
	F_forecast = F_var_model.forecast(F_hat[-lag_order:,:], steps = steps)
	print(f"This is the F forecat \n{F_forecast}, this is the training f \n{F_hat/-0.1710124}")
	return reconstruct_Y(F_forecast, Lambda_hat, Mu_hat)


def reconstruct_Y(F_forecast, Lambda, Mu):
	T = np.shape(F_forecast)[0]
	I = np.shape(Lambda)[0]
	J = np.shape(Mu)[0]
	Y = np.zeros(shape = (I,J,T))
	for i in range(np.shape(F_forecast)[1]):
		Y += np.multiply.outer(np.multiply.outer(Lambda[:,i], Mu[:,i]), F_forecast[:,i])
	return Y


seed = 1#np.random.randint(0,10000)# 990 # 609, 36, 990, 8550,2221# 
print(f"seed = {seed}")
Lags = 1
ar_coeffs = np.array([[0.8]])
n_components = 1
components = n_components*(Lags+1)


I = 3 # Let these be economic time series
J = 3 # These be countries
T = 4+ Lags # These be time periods, 15 years of 4 quarters of data. 
Mu_corr = np.eye(n_components*(Lags+1))#random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Sigma_corr = np.eye(n_components*(Lags+1))#random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Mu_m = np.zeros(shape = n_components*(Lags+1))#np.random.normal(0,1, size = n_components*(Lags+1))
Lambda_m = np.zeros(shape = n_components*(Lags+1))# np.random.normal(0,1, size = n_components*(Lags+1))

Y, M, s = gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda(I,J,T, seed, ar_coefficients = ar_coeffs, noise_scale = 0, y_scale = 5,
        dynamic_factors = n_components, lags = Lags, mu_mean = Mu_m, mu_sigma = Mu_corr, lambda_mean = Lambda_m, lambda_sigma = Sigma_corr, case = 1)

print(f"Np shape Y {np.shape(Y)}")
Lambda = M[0]
Mu = M[1]
F = M[2]


#observed_Y = Y[:,:,:10]

Z = tPCA(Y)
s_hat, M_hat = Z.t_pca(components)
print(f"S_hat {s_hat}")
print(f"{np.shape(np.array([s_hat['0']]))}, {np.shape(M_hat['0'])}")
Lambda_hat = M_hat["0"] #* np.array([s_hat["0"]])
#Lambda_hat = Lambda_hat 
Mu_hat = M_hat["1"] #* s_hat["1"]
F_hat = M_hat["2"] #* s_hat["2"]

print(f"f cos error {aux.get_cos_errors(F, F_hat)}")
print(f"L cos error {aux.get_cos_errors(Lambda, Lambda_hat)}")
print(f"M cos error {aux.get_cos_errors(Mu, Mu_hat)}")

#F_hat = F_hat*s_hat["2"]
#print(f"This is shat {s_hat}")
#print(f"tensorPCA F covariance {F_hat.T @ F_hat}")
#F_var_model, lag_order = fit_var(F_hat)
#F_forecast = F_var_model.forecast(F_hat[-lag_order:,:], steps = 2)
#print(f"This is the F forecat \n{F_forecast}, this is the training f \n{F_hat/-0.1710124}")
#reconstructed_y = reconstruct_Y(F_hat, Lambda_hat, Mu_hat)
#a = np.array([Lambda[[0,1],0]*F[0,0]*Mu[0,0], Mu[0,1]*Lambda[[0,1],1]*F[0,1]]).T
#rot = np.linalg.lstsq(a, Y[0,[0,1],0], rcond = None)[0]
#print(f"Well this is a {a}")
#print(f"This is the rotation to get the values to match as closely as possible {rot} and this is s_hat {s_hat}")

first_bit = np.multiply.outer(np.multiply.outer(Lambda_hat[:,0], Mu_hat[:,0]), F_hat[:,0])
second_bit = np.multiply.outer(np.multiply.outer(Lambda_hat[:,1], Mu_hat[:,1]), F_hat[:,1])

print(f"Lambda norm {np.linalg.norm(Lambda, axis = 0)}, Mu norm {np.linalg.norm(Mu,axis = 0)}, F_norm {np.linalg.norm(F, axis = 0)}")
print(f"True s {s}")
print(f"This is the first bit {first_bit}")
print(f"This is the second bit {second_bit}")
print(f"This is the true Y {Y}")
#print(f"This is the first but plus second bit {first_bit + second_bit}")
# a*-0.00734755 + b* 0.03546013 = -0.02582936
# a* -0.00530416 + b* 0.29295424 =  0.37820788
# a first_bit[0,0,0] + b * second_bit[0,0,0] = Y[0,0,0]
# a*first_bit[0,1,0] + b*second_bit[0,1,0] = Y[0,1,0]
mat = np.array([[first_bit[0,0,0], second_bit[0,0,0]], [first_bit[0,1,0], second_bit[0,1,0]]])
target = np.array([[Y[0,0,0]], [Y[0,1,0]]])
rot = np.linalg.lstsq(mat, target, rcond = None)[0]
print(f"This is the rot {rot[0]*first_bit + rot[1]*second_bit}")

print(f"This is the rot {rot}, the s {s}")
print(f"Distance to true {np.sum((rot[0]*first_bit + rot[1]*second_bit - Y)**2)}")

rot = np.linalg.lstsq(Lambda_hat, Lambda, rcond = None)[0]

#print(f"mat {mat}, target {target}, rot {rot}")
#print(f"Lambda_hat @ rot {Lambda_hat @ rot}")
#print(f"True Lambda {Lambda}")

rot = np.linalg.lstsq(Mu_hat, Mu, rcond = None)[0]

#print(f"Mu_hat @ rot {Mu_hat @ rot}")
#print(f"True Mu {Mu}")

rot = np.linalg.lstsq(F_hat, F, rcond = None)[0]

#print(f"F_hat @ rot {F_hat @ rot}")
#print(f"True F {F}")

#print(f"This is the first and second bit added together {rot[0]*first_bit+rot[1]*second_bit}")
#print(f"This is Y {Y}")

#print(f"Diff btween Y and reconstructed Y {observed_Y-reconstructed_y}")

#tensorPCA_forecast = tensorPCA_predict(Y, 5, components)
#print(f"True F \n{F/-0.21586862}")


		