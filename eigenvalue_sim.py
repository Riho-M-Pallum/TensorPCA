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
from sklearn.decomposition import PCA

cwd = os.getcwd()
images_dir = os.path.join(cwd, "images")
images_subfolder = os.path.join(images_dir, "twoD_eigenvalues")
if not os.path.exists(images_subfolder):
    os.makedirs(images_subfolder)

"""
The goal of this script is to evaluate the eigenvalue separation of my estimator of the number of dynamic factors under different data generating processes.
As a comparison I use Bai and Ng's 2007 paper "Determining the number of Primitive Shocks in Factor models" TO BE IMPLEMENTED


Also I need to check my recover factor for the case of three or more lags. It seems to perform badly there as the number of datapoints goes up.
I think it's more likely that the bad performance is due to my own bad programming abilities.
"""

def simulate_2d_over_IT(simulation_range, I_T_relation, repetitions, n_components, Lags, ar_coeffs, n_singular_values_to_save, noise_scale = 1):
    errors = {"value": [],"rep":[], "some_singular_values": []}
    for value in simulation_range:
        print(value)
        I = value
        T = int(I_T_relation*value)
        for rep in range(repetitions):
            seed = np.random.randint(0,repetitions*1000)
            print("I and seed:")
            print(I, seed)
            errors["value"].append(value)
            errors["rep"].append(rep)
            Y,s,M = DGP_2d_dyn_fac(I,T, n_components, Lags, ar_coeffs, noise_scale, seed = seed)
            Z = tPCA(Y)
            s_hat, M_hat = Z.t_pca(n_components*(Lags+1))
            F_hat = M_hat["1"]
            #pca = PCA(n_components=n_components*(Lags+1))
            #Lambda_hat = pca.fit_transform(Y)
            #F_hat = Y.T @ Lambda_hat @ np.linalg.inv(Lambda_hat.T @ Lambda_hat) 

            # True factors
            F = M[1]
            Lambda = M[0]

            _, S, _ = aux.recover_lags(F_hat, lags = Lags, dynamic_factors = n_components)
            
            errors["some_singular_values"].append(S[-n_singular_values_to_save:])
    return errors

noise_scale = 0

r = range(20, 150, 10)
ratio = 1.1
rep = 250

q = 2
s = 1
ar_coeffs = np.array([[0.8, 0.7]])

err = simulate_2d_over_IT(r, ratio, rep, n_components = q, Lags = s, ar_coeffs = ar_coeffs, n_singular_values_to_save = q*(s+1), noise_scale = noise_scale)
#print(err)
err_df = pd.DataFrame.from_dict(err)
print(err_df.head())
err_df_exploded = err_df.explode("some_singular_values")
err_df_exploded["SV"] = err_df_exploded.groupby(by = ["value", "rep"]).cumcount()   



print(err_df_exploded.head())
print(type(err_df_exploded.iloc[0]["some_singular_values"]))
print(f"Exploded head")
min_by_value = err_df_exploded.groupby(by = ["value","SV"]).min().reset_index()
max_by_value = err_df_exploded.groupby(by = ["value","SV"]).max().reset_index()
mean_by_value = err_df_exploded.groupby(by = ["value","SV"])[["some_singular_values"]].mean().reset_index()


print(max_by_value.head())
print(mean_by_value.head())
print(min_by_value.head())

"""
fig = px.scatter(min_by_value, x = "value", y = "some_singular_values", title = f"Lowest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Lowest_sv_by_value_q={q},s={s}.jpeg"))

fig = px.scatter(max_by_value, x = "value", y = "some_singular_values", title = f"Highest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Highest_sv_by_value_q={q},s={s}.jpeg"))
"""
fig = px.scatter(mean_by_value, x = "value", y = "some_singular_values", title = f"Mean singular value recovered in a rep q = {q}, s = {s}", color = "SV")
fig.show()
fig.write_image(os.path.join(images_dir, f"Mean_sv_by_value_q={q},s={s}.jpeg"))




q = 2
s = 2
ar_coeffs = np.array([[0.8, 0.2], [0.1, 0.5]])

err = simulate_2d_over_IT(r, ratio, rep, n_components = q, Lags = s, ar_coeffs = ar_coeffs, n_singular_values_to_save = q*(s+1), noise_scale = noise_scale)
#print(err)
err_df = pd.DataFrame.from_dict(err)
err_df_exploded = err_df.explode("some_singular_values")
err_df_exploded["SV"] = err_df_exploded.groupby(by = ["value", "rep"]).cumcount()   
print(err_df_exploded.head())


print(err_df_exploded.head())
min_by_value = err_df_exploded.groupby(by = ["value","SV"]).min().reset_index()
max_by_value = err_df_exploded.groupby(by = ["value","SV"]).max().reset_index()
mean_by_value = err_df_exploded.groupby(by = ["value","SV"])[["some_singular_values"]].mean().reset_index()


print(max_by_value.head())
print(mean_by_value.head())
print(min_by_value.head())

"""
fig = px.scatter(min_by_value, x = "value", y = "some_singular_values", title = f"Lowest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Lowest_sv_by_value_q={q},s={s}.jpeg"))

fig = px.scatter(max_by_value, x = "value", y = "some_singular_values", title = f"Highest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Highest_sv_by_value_q={q},s={s}.jpeg"))
"""
fig = px.scatter(mean_by_value, x = "value", y = "some_singular_values", title = f"Mean singular value recovered in a rep q = {q}, s = {s}", color = "SV")
fig.show()
fig.write_image(os.path.join(images_dir, f"Mean_sv_by_value_q={q},s={s}.jpeg"))


q = 3
s = 2
ar_coeffs = np.array([[0.8, 0.2, 0.7], [0.1, 0.2, 0.3]])

err = simulate_2d_over_IT(r, ratio, rep, n_components = q, Lags = s, ar_coeffs = ar_coeffs, n_singular_values_to_save = q*(s+1), noise_scale = noise_scale)
#print(err)
err_df = pd.DataFrame.from_dict(err)
print(err_df.head())
err_df_exploded = err_df.explode("some_singular_values")
err_df_exploded["SV"] = err_df_exploded.groupby(by = ["value", "rep"]).cumcount()   



print(err_df_exploded.head())
min_by_value = err_df_exploded.groupby(by = ["value","SV"]).min().reset_index()
max_by_value = err_df_exploded.groupby(by = ["value","SV"]).max().reset_index()
mean_by_value = err_df_exploded.groupby(by = ["value","SV"])[["some_singular_values"]].mean().reset_index()


print(max_by_value.head())
print(mean_by_value.head())
print(min_by_value.head())

"""
fig = px.scatter(min_by_value, x = "value", y = "some_singular_values", title = f"Lowest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Lowest_sv_by_value_q={q},s={s}.jpeg"))

fig = px.scatter(max_by_value, x = "value", y = "some_singular_values", title = f"Highest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Highest_sv_by_value_q={q},s={s}.jpeg"))
"""
fig = px.scatter(mean_by_value, x = "value", y = "some_singular_values", title = f"Mean singular value recovered in a rep q = {q}, s = {s}", color = "SV")
fig.show()
fig.write_image(os.path.join(images_dir, f"Mean_sv_by_value_q={q},s={s}.jpeg"))



q = 1
s = 3
ar_coeffs = np.array([[0.8], [0.5], [0.1]])

err = simulate_2d_over_IT(r, ratio, rep, n_components = q, Lags = s, ar_coeffs = ar_coeffs, n_singular_values_to_save = q*(s+1), noise_scale = noise_scale)
#print(err)
err_df = pd.DataFrame.from_dict(err)
print(err_df.head())
err_df_exploded = err_df.explode("some_singular_values")
err_df_exploded["SV"] = err_df_exploded.groupby(by = ["value", "rep"]).cumcount()   



print(err_df_exploded.head())
min_by_value = err_df_exploded.groupby(by = ["value","SV"]).min().reset_index()
max_by_value = err_df_exploded.groupby(by = ["value","SV"]).max().reset_index()
mean_by_value = err_df_exploded.groupby(by = ["value","SV"])[["some_singular_values"]].mean().reset_index()


print(max_by_value.head())
print(mean_by_value.head())
print(min_by_value.head())

"""
fig = px.scatter(min_by_value, x = "value", y = "some_singular_values", title = f"Lowest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Lowest_sv_by_value_q={q},s={s}.jpeg"))

fig = px.scatter(max_by_value, x = "value", y = "some_singular_values", title = f"Highest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Highest_sv_by_value_q={q},s={s}.jpeg"))
"""
fig = px.scatter(mean_by_value, x = "value", y = "some_singular_values", title = f"Mean singular value recovered in a rep q = {q}, s = {s}", color = "SV")
fig.show()
fig.write_image(os.path.join(images_dir, f"Mean_sv_by_value_q={q},s={s}.jpeg"))





q = 2
s = 3
ar_coeffs = np.array([[0.8, 0.7], [0.5, 0.2], [0.1, 0]])

err = simulate_2d_over_IT(r, ratio, rep, n_components = q, Lags = s, ar_coeffs = ar_coeffs, n_singular_values_to_save = q*(s+1), noise_scale = noise_scale)
#print(err)
err_df = pd.DataFrame.from_dict(err)
print(err_df.head())
err_df_exploded = err_df.explode("some_singular_values")
err_df_exploded["SV"] = err_df_exploded.groupby(by = ["value", "rep"]).cumcount()   



print(err_df_exploded.head())
min_by_value = err_df_exploded.groupby(by = ["value","SV"]).min().reset_index()
max_by_value = err_df_exploded.groupby(by = ["value","SV"]).max().reset_index()
mean_by_value = err_df_exploded.groupby(by = ["value","SV"])[["some_singular_values"]].mean().reset_index()


print(max_by_value.head())
print(mean_by_value.head())
print(min_by_value.head())


"""fig = px.scatter(min_by_value, x = "value", y = "some_singular_values", title = f"Lowest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Lowest_sv_by_value_q={q},s={s}.jpeg"))

fig = px.scatter(max_by_value, x = "value", y = "some_singular_values", title = f"Highest singular value recovered in a rep q = {q}, s = {s}", color = "SV")
#fig.show()
fig.write_image(os.path.join(images_dir, f"Highest_sv_by_value_q={q},s={s}.jpeg"))

"""
fig = px.scatter(mean_by_value, x = "value", y = "some_singular_values", title = f"Mean singular value recovered in a rep q = {q}, s = {s}", color = "SV")
fig.show()
fig.write_image(os.path.join(images_dir, f"Mean_sv_by_value_q={q},s={s}.jpeg"))

