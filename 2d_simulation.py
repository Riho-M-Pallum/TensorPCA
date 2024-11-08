import numpy as np
import pandas as pd
from TensorPCA.tensorpca import TensorPCA as tPCA
from TensorPCA.dgp import DGP
from TensorPCA.dgp import DGP_2d_dyn_fac
from TensorPCA.dgp import gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import TensorPCA.auxiliary_functions as aux
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

cwd = os.get_cwd()
images_dir = os.path.join(cwd, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Ugly simulation, I should put the entre process into a class but oh well

def simulate_2d_over_IT(simulation_range, I_T_relation, repetitions, n_components, Lags, ar_coeffs, noise_scale = 1):
	errors = {"value": [],"rep":[], "F_cos_error":[], "F_R2_error":[], "Lambda_cos_error":[], "Lambda_R2_error":[],
			 "recovered_R2_error":[], "recovered_cos_error":[], "some_singular_values": [], "F_timed_error":[], "recovered_L21_error":[]}
	for value in simulation_range:
		print(value)
		I = value
		T = int(I_T_relation*value)
		for rep in range(repetitions):
			seed = np.random.randint(0,repetitions*1000)
			errors["value"].append(value)
			errors["rep"].append(rep)
			Y,s,M = DGP_2d_dyn_fac(I,T, n_components, Lags, ar_coeffs, noise_scale, seed = seed)
			Z = tPCA(Y)
			s_hat, M_hat = Z.t_pca(n_components*(Lags+1))
			# True factors

			F = M[1]
			Lambda = M[0]

			F_hat = M_hat["1"]
			Lambda_hat = M_hat["0"]
			errors["F_cos_error"].append(aux.get_cos_errors(F, F_hat))
			errors["F_R2_error"].append(aux.get_R2_errors(F, F_hat))
			errors["Lambda_cos_error"].append(aux.get_cos_errors(Lambda, Lambda_hat))
			errors["Lambda_R2_error"].append(aux.get_R2_errors(Lambda, Lambda_hat))
			
			recovered, S = aux.recover_lags(F_hat, lags = Lags, dynamic_factors = n_components)
			recovered = recovered[:,[1,3,0,2]]
			cont_recovered = recovered[:,[0,1]]
			lagged_recovered = recovered[:,[2,3]]
			cont_vars = F[:,[0,1]]
			lagged_vars = F[:,[2,3]]
			cont_errors = aux.get_cos_errors(cont_vars, cont_recovered)
			lagg_errors = aux.get_cos_errors(lagged_vars, lagged_recovered)
			errors["some_singular_values"].append(S[-n_components*(Lags+1):])
			errors["recovered_cos_error"].append(aux.get_cos_errors(F, recovered))
			errors["recovered_R2_error"].append(aux.get_R2_errors(F, recovered))
			errors["recovered_L21_error"].append(aux.get_L21_errors(F, recovered/np.linalg.norm(recovered, axis = 0)))
			errors["F_timed_error"].append(np.concatenate((cont_errors, lagg_errors)))
	return errors


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


r = range(20, 230, 10)
ratio = 1.1
rep = 250

ar_coeffs = np.array([[0.8, 0.2]])

err = simulate_2d_over_IT(r, ratio, rep, 2, 1, ar_coeffs, noise_scale = .5)
#print(err)
err_df = pd.DataFrame.from_dict(err)
	
err_df_exploded = err_df.explode(list(err_df.columns[2:-1])).drop(columns = ["recovered_L21_error"])
err_df_exploded["factor"] = err_df_exploded.groupby(by = ["value", "rep"]).cumcount()	



print(err_df_exploded.head())
min_by_value = err_df_exploded.groupby(by = ["value","factor"]).min().reset_index()
max_by_value = err_df_exploded.groupby(by = ["value","factor"]).max().reset_index()
mean_by_value = err_df_exploded.groupby(by = ["value","factor"])[list(err_df_exploded.columns[2:-1])].mean().reset_index()


print(max_by_value.head())
print(mean_by_value.head())
print(min_by_value.head())


fig = px.scatter(min_by_value.query("factor == 3"), x = "value", y = "recovered_cos_error", title = "Worst cos error")
fig.add_trace(go.Scatter(x = mean_by_value.query("factor == 3")["value"], y = mean_by_value.query("factor == 3")["recovered_cos_error"]))
fig.show()
fig.write_image(os.path.join(images_dir, "Worst_cos_error.jpeg"))

fig = px.scatter(mean_by_value.query("factor == 3"), x = "value", y = "F_cos_error", title = "F vs recovered cos error")
fig.add_trace(go.Scatter(x = min_by_value.query("factor == 3")["value"], y = min_by_value.query("factor == 3")["recovered_cos_error"]))
fig.show()
fig.write_image(os.path.join(images_dir, "F_vs_rotated_F_error.jpeg"))

singular_values_averages = err_df_exploded.groupby(by = ["value","factor"])[["some_singular_values"]].mean().reset_index()
singular_values_averages["factor"] = singular_values_averages["factor"].astype(str)
print(singular_values_averages)
fig = px.scatter(singular_values_averages, x = "value", y = "some_singular_values", color = "factor", title = "Average 4 smallest singular values of restriction matrix")
fig.show()	
fig.write_image(os.path.join(images_dir, "Average_4_smallest_sv_of_restriction.jpeg"))


fig = px.scatter(mean_by_value, x = "value", y = "F_timed_error", color = "factor", title = "Error by time, 0,1 = contemporaneous; 2,3 = first lag")
fig.show()
fig.write_image(os.path.join(images_dir, "Error_by_time_contemporaneous_vs_lag.jpeg"))

fig = px.scatter(mean_by_value, x = "value", y = "F_cos_error", color = "factor", title = "Average F cos error by factor")
fig.show()
fig.write_image(os.path.join(images_dir, "Average_cos_error_by_factor.jpeg"))


