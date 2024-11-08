import pandas as pd
import numpy as np
import os
from TensorPCA.tensorpca import TensorPCA as tPCA
import TensorPCA.auxiliary_functions as aux
import plotly.express as px

cwd = os.getcwd()
data_folder = os.path.join(cwd, "FREDMD")
os.chdir(data_folder)
df = pd.read_csv("current.csv", index_col = 0, parse_dates = True,  header=  0)
transform = df.loc["Transform:"].to_frame(name = "transform").reset_index().rename(columns = {"index":"series"})
df_long = pd.melt(df.drop("Transform:").reset_index(), id_vars = ["sasdate"], var_name = "series", value_name = "value")
df_long = pd.merge(df_long, transform, how = "left", on = "series")
df_long["sasdate"] = pd.to_datetime(df_long["sasdate"], format = "%m/%d/%Y")

transform_dict = {
		"1": lambda x: x,
		"2": lambda x: x.diff(),
		"5": lambda x: np.log(x).diff(),
		"6": lambda x: np.log(x).diff().diff()
}
def apply_transform(group):
    transform_type = group['transform'].iloc[0]  # Get the transformation identifier
    # Retrieve the corresponding function from the dictionary
    func = transform_dict.get(f"{transform_type}", lambda x: np.nan)  # Default to NaN if unknown

    group['transformed_value'] = func(group['value'])
    return group

def calc_IC(true, estimated, k):
	residuals_sum = np.sum((true-estimated)**2)
	prod = np.prod(np.shape(true))
	summ = np.sum(np.shape(true))
	IC = np.log(residuals_sum/prod) + k * summ/prod * np.log(prod/summ)
	return IC

def calc_resid(true, estimated):
	residuals_sum = np.sum((true-estimated)**2)
	prod = np.prod(np.shape(true))
	return residuals_sum/prod

df_long = df_long.sort_values(by = ["series", "sasdate"])
df_long["transform"] = df_long["transform"].astype(int)
df_transformed = df_long.groupby(by = ["series"]).apply(apply_transform)

# Use Jauary 1960 to December 1984 to predict January 1985 to December 1985
# Repeat adding data up to January 1985 to predict from February 1985 to January 1986. Repeat until Dec 2022.

start_date = pd.to_datetime("01/01/1960", format = "%m/%d/%Y")
temp_date = pd.to_datetime("01/12/1984", format = "%m/%d/%Y")
end_date_list = pd.date_range(start="01/12/1984", end="01/12/2022", freq='M').to_list()

df_var = df_transformed.pivot(index = "sasdate", columns = "series", values = "transformed_value")
df_var = df_var.loc[start_date:temp_date]
# Drop everything that has any na values
df_var = df_var.dropna(axis = 1, how = "any")
df_var_values = df_var.values

# Rows = time
print(df_var)

print(f"True Y shape {np.shape(df_var.values)}")
#print(df_var.values)


n_components = 1

for n_components in range(1,10):
	Z = tPCA(df_var_values)
	s_hat, M_hat = Z.t_pca(n_components)

	F_hat = M_hat["0"]
	Lambda_hat = M_hat["1"]

	print(f"F hat shape {np.shape(F_hat)}, Lambda hat shape {np.shape(Lambda_hat)}")
	Y_hat = F_hat @ Lambda_hat.T
	IC = calc_IC(df_var_values, Y_hat, n_components)

	print(f"IC for {n_components} factors: {IC}")

# Ok so it doesn't really get any better after 1 factor. Nevertheless let' plug in four factors and see whether my normalisation
# Has any effect and what the singular values look like

n_components = 2

Z = tPCA(df_var_values)
s_hat, M_hat = Z.t_pca(n_components)

F_hat = M_hat["0"]
Lambda_hat = M_hat["1"]

Y_hat = F_hat @ Lambda_hat.T
#IC = calc_IC(df_var_values, Y_hat, n_components)

recovered,S,rot = aux.recover_lags(F_hat, lags = 1, dynamic_factors = 1)
#print(f"These are the eigenvalues of the restriction matrix \n{S}")
#print(f"IC for {n_components} factors: {IC}")
print(f"These are the rot matrices {rot}")
print(f"And these are the singular values of the restriction matrix \n{S}")
d1, d2 = aux.Bai_Ng_determine_q(F_hat, 1)
print(f"This is Bai and Ng criteria 1 \n{d1}, this is their criteria 2 \n{d2}")

# Ok nice, we get that when the Bai and Ng criteria believe there to be a dynamic factor, then my restriction also point that wayx
print(df_var.head())
temp = df_transformed.pivot(index = "sasdate", columns = "series", values = "value")

fig = px.scatter(df_var, x = df_var.index, y ="AAA")
fig.show()
fig = px.scatter(temp, x = temp.index, y ="AAA")
fig.show()
