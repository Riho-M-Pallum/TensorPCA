import numpy as np
import pandas as pd
import datetime as dt
import DynamicFactorAnalysis.dynamicfactoranalysis as dfa
from TensorPCA.tensorpca import TensorPCA as tPCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from TensorPCA.dgp import DGP_2d_dyn_fac
import TensorPCA.auxiliary_functions as aux
from plotly.subplots import make_subplots
import os

cwd = os.getcwd()
images_dir = os.path.join(cwd, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
# Again very ugly simulation, rewrite into a class if I am serious about this
def simulate_2d_over_IT(simulation_range, I_T_relation, repetitions, n_components, Lags, ar_coeffs, noise_scale = 1):
	errors = {"value": [],"rep":[], "F_PCA_error": [], "My_cos_error":[], "My_timed_error":[],
				"FHLR_cos_error": [], "FHLR_timed_error":[]}
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
			errors["F_PCA_error"].append(aux.get_cos_errors(F, F_hat))
			recovered, S, _ = aux.recover_lags(F_hat, lags = Lags, dynamic_factors = n_components)
			recovered = recovered[:,[1,3,0,2]]
			cont_recovered = recovered[:,[0,1]]
			lagged_recovered = recovered[:,[2,3]]
			cont_vars = F[:,[0,1]]
			lagged_vars = F[:,[2,3]]
			cont_errors = aux.get_cos_errors(cont_vars, cont_recovered)
			lagg_errors = aux.get_cos_errors(lagged_vars, lagged_recovered)
			errors["My_cos_error"].append(aux.get_cos_errors(F, recovered))
			errors["My_timed_error"].append(np.concatenate((cont_errors, lagg_errors)))

			# Now for my implementation of FHLR
			M, cov, idio = aux.FHLR_2D_estimation(Y, n_components, Lags, H = 100, M = int(np.sqrt(T)))
			F_hat = M[1]
			Lambda_hat = M[0]
			errors["FHLR_cos_error"].append(aux.get_cos_errors(F, F_hat))
			# Let's also see what happens if we rotate them
			recovered, S, _ = aux.recover_lags(F_hat, lags = Lags, dynamic_factors = n_components)
			recovered = recovered[:,[1,3,0,2]]
			cont_recovered = recovered[:,[0,1]]
			lagged_recovered = recovered[:,[2,3]]
			cont_vars = F[:,[0,1]]
			lagged_vars = F[:,[2,3]]
			cont_errors = aux.get_cos_errors(cont_vars, cont_recovered)
			lagg_errors = aux.get_cos_errors(lagged_vars, lagged_recovered)
			errors["FHLR_timed_error"].append(np.concatenate((cont_errors, lagg_errors)))
			


	return errors

r = range(20, 200, 10)
ratio = 1.1
rep = 100

ar_coeffs = np.array([[0.8, 0.2]])

err = simulate_2d_over_IT(r, ratio, rep, 2, 1, ar_coeffs, noise_scale = .5)
#print(err)
err_df = pd.DataFrame.from_dict(err)
	
err_df_exploded = err_df.explode(list(err_df.columns[2:]))
err_df_exploded["factor"] = err_df_exploded.groupby(by = ["value", "rep"]).cumcount()	



min_by_value = err_df_exploded.groupby(by = ["value","factor"]).min().reset_index()
max_by_value = err_df_exploded.groupby(by = ["value","factor"]).max().reset_index()
mean_by_value = err_df_exploded.groupby(by = ["value","factor"])[list(err_df_exploded.columns[2:-1])].mean().reset_index()


print(max_by_value.head())
print(mean_by_value.head())
print(min_by_value.head())


fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)


fig.add_trace(go.Scatter(x = mean_by_value["value"], y = mean_by_value["FHLR_timed_error"], mode = "lines+markers",  marker=dict(
            size=12,
            color=mean_by_value["factor"],  # Set color based on 'color' column
            colorscale='Viridis',  # Choose a color scale
            showscale=False  # Display color bar
        ), name = "FHLR timed error by factor"))

fig.add_trace(go.Scatter(x = mean_by_value["value"], y = mean_by_value["My_timed_error"], mode = "lines+markers",  marker=dict(
            size=12,
            color=mean_by_value["factor"],  # Set color based on 'color' column
            colorscale='Viridis',  # Choose a color scale
            showscale=False  # Display color bar
        ), name = "My timed error by factor"))

fig.show()
fig.write_image(os.path.join(images_dir, "PCA_rotated_vs_FHLR_rotated_cos_error.jpeg"))



fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
fig.add_trace(go.Scatter(x = mean_by_value["value"], y = mean_by_value["F_PCA_error"], mode = "lines+markers",  marker=dict(
            size=12,
            color=mean_by_value["factor"],  # Set color based on 'color' column
            colorscale='Viridis',  # Choose a color scale
            showscale=True  # Display color bar
        ), name = "F PCA cos error by factor"))

fig.add_trace(go.Scatter(x = mean_by_value["value"], y = mean_by_value["FHLR_cos_error"], mode = "lines+markers",  marker=dict(
            size=12,
            color=mean_by_value["factor"],  # Set color based on 'color' column
            colorscale='Viridis',  # Choose a color scale
            showscale=True  # Display color bar
        ), name = "FHLR cos error by factor"))

fig.add_trace(go.Scatter(x = mean_by_value["value"], y = mean_by_value["My_cos_error"], mode = "lines+markers",  marker=dict(
            size=12,
            color=mean_by_value["factor"],  # Set color based on 'color' column
            colorscale='Viridis',  # Choose a color scale
            showscale=True  # Display color bar
        ), name = "My cos error by factor"))

fig.show()
fig.write_image(os.path.join(images_dir, "Rotated_vs_PCA_vs_FHLRunrotated_cos_error.jpeg"))
