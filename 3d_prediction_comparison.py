import numpy as np
import pandas as pd
from TensorPCA.tensorpca import TensorPCA as tPCA
from TensorPCA.dgp import DGP
from TensorPCA.dgp import DGP_2d_dyn_fac
from TensorPCA.dgp import gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda
from statsmodels.tsa.ar_model import AutoReg
#import matplotlib.pyplot as plt
import TensorPCA.auxiliary_functions as aux
import statsmodels.tsa.api as sm
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import random_correlation


""" 
The way that I carry out prediction is that I generate a model of say size (I,N,T), I want to predict new entries in the T direction,
 so for finding the loadings I only use floor( T*x ), x in (0,1) observations. With the final 1-T*x observations serving as the set that I wish
 to predict over. Prediction requires two steps. First simulating the path of the factors using an estimated AR model, then plugging those into the 
 equation for y at a future date.	
 Naturally if y_{t} = Lambda1 * F_{t} + Lambda_2 * F_{t-1} + e_{t}
 Then E[y_{t+1}] = Lambda1 E[F_{t+1}] + Lambda_2 F_{t} + E[e_{t+1}] = Lambda1 AF_{t} + Lambda2 F_{t}

 In the same way 
 E[y_{t+h}] = Lambda1E[f_{t+h}] + lambda2 E[F_{t+h-1}] = Lambda1 A^{h}f_{t} + Lambda2 A^{h-1}f_{t}?


When we do prediction, we want to compare multiple different appraoches. The approaches implemented are

1) TensorPCA factor model
2) Unfold + VAR 
3) Unfold + factor model
4) TensorPCA with factor VAR estimated on the rotated factors

We carry out this exercise for two broadly different settings. In the first setting we look at forecasting the entire
data series. In the second setting we look only at forecasting a subset of the data.

So when doing the actual prediction. When I rotate F, I must also rotate Lambda so that they match and I get back the true data.
So way 1) of doing things would be:
a) TensorPCA for span
b) Rotate resulting F vectors
c) Rotate Lambda/Mu accordingly
d) Predict next value for F
e) Multiply everything back together for prediction of Y

Way 2) of doing things would be:
a) TensorPCA for span
b) rotate resulting F vector
c) predict next entry in F vector
d) rotate F vector back
e) Multiply everything back togethr for prediction of Y

In this script I will follow way 2 for no particular reason other than the fact that I worked out the rotation for F first.

"""
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


def reconstruct_Y(F_forecast, Lambda, Mu):
	T = np.shape(F_forecast)[0]
	I = np.shape(Lambda)[0]
	J = np.shape(Mu)[0]
	Y = np.zeros(shape = (I,J,T))
	for i in range(np.shape(F_forecast)[1]):
		Y += np.multiply.outer(np.multiply.outer(Lambda[:,i], Mu[:,i]), F_forecast[:,i])
	return Y


def tensorPCA_predict(Y, steps, components):
	Z = tPCA(Y)
	s_hat, M_hat = Z.t_pca(components)
	Mu_hat = M_hat["0"]
	Lambda_hat = M_hat["1"]
	F_hat = M_hat["2"]
	print(f"tensorPCA F covariance {F_hat.T @ F_hat}")
	Mu_hat, Lambda_hat, F_hat = aux.alternating_least_squares(Y, F_hat, Lambda_hat, Mu_hat, 0.0001) # M, L, F -> 5,4,3 | L,M,F -> 5,3,4 | M,F,L -> 4,5,3
	print(f"Shape of F hat after alternating least squares {np.shape(F_hat)}")
	F_var_model, lag_order = fit_var(F_hat)
	F_forecast = F_var_model.forecast(F_hat[-lag_order:,:], steps = steps)
	return reconstruct_Y(F_forecast, Lambda_hat, Mu_hat)

def direct_var_forecast(Y, steps):
	"""
	Assumes that the time dimension is the third dimension.
	Also note that we can often have too many parameters to estimate a VAR. So to deal with that I have to cut down
	on the number of observations that I feed into the model
	"""
	F_var_model, lag_order = fit_var(Y)
	forecast = F_var_model.forecast(Y[-lag_order:,:], steps = steps)
	return forecast

def direct_factor_forecast(Y, steps, components):
	Z = tPCA(Y)
	s_hat, M_hat = Z.t_pca(components)

	F_hat = M_hat["1"]
	Lambda_hat = M_hat["0"]
	print(f"Shape of F_hat = {np.shape(F_hat)}, with covariance {F_hat.T @ F_hat}")
	F_var_model, lag_order = fit_var(F_hat)
	F_forecast = F_var_model.forecast(F_hat[-lag_order:,:], steps = steps)
	print(f"Shape of lambda_hat {np.shape(Lambda_hat)}, shape of F forecast {np.shape(F_forecast)}")
	return Lambda_hat @ F_forecast.T

def tensorPCA_rotate(Y, steps, dynamic_factors, lags):
	Z = tPCA(Y)
	s_hat, M_hat = Z.t_pca(dynamic_factors*(1+lags))
	Lambda_hat = M_hat["0"]
	Mu_hat = M_hat["1"]
	F_hat = M_hat["2"]
	F_rotated , S, rot_matrix = aux.recover_lags(F_hat, lags = lags, dynamic_factors = dynamic_factors)
	rot_inv = np.linalg.inv(rot_matrix.T)

	F_var_model, lag_order = fit_var(F_hat)
	forecast = F_var_model.forecast(F_hat[-lag_order:,:], steps = steps)

	rotated_back_F = forecast @ rot_inv

	return reconstruct_Y(rotated_back_F, Lambda_hat, Mu_hat) 

def simulate(I,J,T, observed_T, n_components, lags, ar_coeffs, reps):
	forecast_errors = {
			"var_error": np.array([]), "factor_error": np.array([]),
			"tensorPCA_error": np.array([]),
			"horizon":np.array([]),
			"rep":np.array([])
		}

	for rep in range(reps):
		Mu_corr = np.eye(n_components*(Lags+1))#random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
		Sigma_corr = np.eye(n_components*(Lags+1))#random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
		Mu_m = np.zeros(shape = n_components*(Lags+1))#np.random.normal(0,1, size = n_components*(Lags+1))
		Lambda_m = np.zeros(shape = n_components*(Lags+1))# np.random.normal(0,1, size = n_components*(Lags+1))

		Y, M, _ = gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda(I,J,T, seed, ar_coefficients = ar_coeffs, noise_scale = 0, y_scale = 5,
		        dynamic_factors = n_components, lags = Lags, mu_mean = Mu_m, mu_sigma = Mu_corr, lambda_mean = Lambda_m, lambda_sigma = Sigma_corr, case = 1)
		print(f"True Y shape {np.shape(Y)}")
		Lambda = M[0]
		Mu = M[1]
		F = M[2]
		forecast_steps = T-observed_T-1
		# Now let's focus on predicting a single country and the 10 time series in said country
		# Say that we observe only 13 years of data and we want to predict up to two years into the future.
		observed_Y = Y[:,:,:observed_T]

		single_country = Y[:,0,:observed_T]
		true_single_country = Y[:,0,observed_T:]
		print(f"true_single_country {true_single_country}")
		print(f"np.shape(true_single_country){np.shape(true_single_country)}")
		
		# First the VAR forecast
		# Problems with the data being non stationary
		var_forecast = direct_var_forecast(single_country.T, forecast_steps).T #(n_samples, n_variables)

		# Then the usual tensor PCA forecast

		tensorPCA_forecast = tensorPCA_predict(observed_Y, forecast_steps, components)

		# Then the direct factor model forecast

		factor_forecast = direct_factor_forecast(single_country, forecast_steps, components)

		# Then the rotated tensor PCA forecast

		#rotated_tensorPCA_forecast = tensorPCA_rotate(observed_Y, forecast_steps, n_components, Lags)

		var_forecast_error = np.sum((true_single_country - var_forecast)**2, axis = 0)
		factor_forecast_error = np.sum((true_single_country - factor_forecast)**2, axis = 0)

		tensorPCA_forecast_error = np.sum((true_single_country - tensorPCA_forecast[:,0,:])**2, axis = 0)
		
		forecast_errors["var_error"] = np.concatenate((forecast_errors["var_error"], var_forecast_error))
		forecast_errors["factor_error"] = np.concatenate((forecast_errors["factor_error"], factor_forecast_error))
		forecast_errors["tensorPCA_error"] = np.concatenate((forecast_errors["tensorPCA_error"], tensorPCA_forecast_error))
		forecast_errors["horizon"] = np.concatenate((forecast_errors["horizon"], [i for i in range(1, forecast_steps+1)]))
		forecast_errors["rep"] = np.concatenate((forecast_errors["rep"], [rep for _ in range(forecast_steps)]))

		
	error_df = error_df = pd.DataFrame.from_dict(forecast_errors)
	error_df = pd.melt(error_df,id_vars = ["horizon","rep"], value_vars = error_df.columns[:-2])

	print(error_df.head())
	error_df = error_df.groupby(by = ["horizon","variable"]).mean().reset_index()
	print(error_df.head())
	
	return error_df

	


seed = np.random.randint(0,10000)# 990 # 609, 36, 990, 8550,2221# 
print(f"seed = {seed}")
Lags = 1
ar_coeffs = np.array([[0.8]])
n_components = 1
components = n_components*(Lags+1)


I = 5 # Let these be economic time series
J = 5 # These be countries
T = 15*4 + Lags # These be time periods, 15 years of 4 quarters of data. 

error_df = simulate(I,J,T, 13*4, n_components = n_components, lags = Lags, ar_coeffs = ar_coeffs, reps = 1)
fig = px.scatter(error_df, x = "horizon", y = "value", color = "variable")
fig.data[0].update(mode='markers+lines')
fig.data[1].update(mode='markers+lines')
fig.data[2].update(mode='markers+lines')

fig.show()

aaaa































# So if we increase the time dimension too much with respect to the others then we do not recover all the factors well
# Something weird is going on, sometimes we recover the last two factors well and sometimes we recover them really badly, seems to be seed dependent.
# That is most likely going to be small sample problems.


print(f"this is the tensorPCA forecast {np.shape(tensorPCA_forecast)}, this is the rotated tensorPCA forecat {np.shape(rotated_tensorPCA_forecast)}")
print(f"This is the direct factor_forecasst {np.shape(factor_forecast)} and the var {np.shape(var_forecast)}")



# Now to calculate how close they were to the true data
# In calculating this, I want to get the difference by period. I epxect the forecasts to be better for the current period
# and gradually decay as the forecast horizon increases
# As the first benchmark I just take RMSE error. I should probably compare to an AR(1) benchmark
var_forecast_error = np.sum((true_single_country - var_forecast)**2, axis = 0)
factor_forecast_error = np.sum((true_single_country - factor_forecast)**2, axis = 0)

tensorPCA_forecast_error = np.sum((true_single_country - tensorPCA_forecast[:,0,:])**2, axis = 0)
rotated_tensorPCA_forecast_error = np.sum((true_single_country - rotated_tensorPCA_forecast[:,0,:])**2, axis = 0)

forecast_errors = {
	"var_error": var_forecast_error, "factor_error": factor_forecast_error,
	"tensorPCA_error": tensorPCA_forecast_error, "rotated_tensorPCA_error": rotated_tensorPCA_forecast_error,
	"horizon":[i for i in range(1,forecast_steps+1)]
}
for key in forecast_errors.keys():
	print(f"Shape of {key} {np.shape(forecast_errors[key])}")

error_df = pd.DataFrame.from_dict(forecast_errors)
error_df = pd.melt(error_df,id_vars = ["horizon"], value_vars = error_df.columns[:-1])
print(error_df.head())

fig = px.scatter(error_df, x = "horizon", y = "value", color = "variable")
fig.show()
