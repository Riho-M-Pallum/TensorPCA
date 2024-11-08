import numpy as np
import pandas as pd
from TensorPCA.tensorpca import TensorPCA as tPCA
from TensorPCA.dgp import DGP
from TensorPCA.dgp import DGP_2d_dyn_fac
from TensorPCA.dgp import gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda
from statsmodels.tsa.ar_model import AutoReg
import plotly.graph_objs as go
import TensorPCA.auxiliary_functions as aux
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
from sklearn.decomposition import PCA

seed = 508#np.random.randint(0,10000)# 990 # 609, 36, 990, 8550,2221# 
print(f"seed = {seed}")
Lags = 3
ar_coeffs = np.array([[0.8], [0.5], [0.1]])
n_components = 1
I = 140
T = int(140*1.1) + Lags



	
Y, s, M = DGP_2d_dyn_fac(I,T, n_components, Lags, ar_coeffs, noise_scale = 0.5, seed = seed)

print(f"The shape of Y {np.shape(Y)}")

print(s)

Z = tPCA(Y)
s_hat, M_hat = Z.t_pca(n_components*(Lags+1))
# True factors

F = M[1]
Lambda = M[0]

pca = PCA(n_components=n_components*(Lags+1))
loadings = pca.fit_transform(Y)
factors = Y.T @ loadings @ np.linalg.inv(loadings.T @ loadings) 

F_hat = M_hat["1"]
Lambda_hat = M_hat["0"]
print(f"We have Lambda has shape {np.shape(M[0])}, Mu has shape {np.shape(M[1])}")
"""
print("F R2 error")
print(aux.get_R2_errors(F, F_hat))
print("F cos error")
print(aux.get_cos_errors(F, F_hat))
print(f"Direct PCA F cos error {aux.get_cos_errors(F, factors)}")
print("F L21 error")
print(aux.get_L21_errors(F, F_hat))

print("Lambda R2 error")
print(aux.get_R2_errors(Lambda, Lambda_hat))
print("Lambda cos error")
print(aux.get_cos_errors(Lambda, Lambda_hat))
"""
print("Now to recover a rotaton")
# Basically I am recovering the rotation with a normalised vector
recovered,S,_ = aux.recover_lags(F_hat, lags = Lags, dynamic_factors = n_components)

print(f"These are the eigenvalues of the restriction matrix \n{S}")

"""
#print(recovered)
print(f"recovered rotation R2 error \n{aux.get_R2_errors(F, recovered)}")
print(f"recovered rotation cos error \n{aux.get_cos_errors(F, recovered)}")
print("normalised recovered L21 error")
print(aux.get_L21_errors(F, recovered/np.linalg.norm(recovered, axis = 0)))

print(f"Whether the recovered rotation falls into the original recovered \n{aux.get_cos_errors(F_hat, recovered)}")

cont_vars = F[:,[0,1]]
cont_recovered = recovered[:,[1,3]]
print(f"Contemporaneous projection error {aux.get_cos_errors(cont_vars, cont_recovered)}")
#Ok we have (x,y)(a,b).T = c, find a and b
sol = np.linalg.lstsq(cont_recovered, cont_vars, rcond = None)
print(f"True rotation is \n{sol[0]}")
print(f"The shapes {np.shape(sol[0])}, the vectors {np.shape(cont_recovered)}")
#print(f"L21 error between true and rotated recovered {aux.get_L21_errors(cont_vars, cont_recovered@sol[0])}")
print(f"Smallest 3 eigenvalues {S[-3:]}")
#print(recovered)
aa
#print(f"This is the recovered rotated to match the true \n{cont_recovered@sol[0]}")
#print(f"This is the actual F\n{F}")
#print(recovered)
"""

#print(f"This is F\n{F}")

"""
model = sm.tsa.AutoReg(F[:,0], lags=1)  # AR(1) model
result = model.fit()
print(result.summary())
model = sm.tsa.AutoReg(F[:,1], lags=1)  # AR(1) model
result = model.fit()
print(result.summary())

#plt.show()
#model = sm.tsa.arima.ARIMA(cont_recovered[:,1], order = (1,0,1))  # ARMA(2,1) model
#result = model.fit()
#print(result.summary())
#print(f"We want the coefficients to be {sol[0][:,0]@ar_coeffs.T} and {np.prod(ar_coeffs)}")


acf_values = st.acf(F[:,0], nlags=30)
pacf_values = st.pacf(F[:,0], nlags = 30)
acf_plot = go.Figure(data=go.Bar(x=np.arange(len(acf_values)), y = acf_values))
acf_plot.add_trace(go.Bar(x=np.arange(len(pacf_values)), y = pacf_values))

acf_plot.update_layout(
    title="True Autocorrelation and Partial Autocorrelation Function (ACF/PACF)",
    xaxis_title="Lags",
    yaxis_title="ACF/PACF",
    showlegend=False
)

acf_plot.show()


acf_values = st.acf(cont_recovered[:,1], nlags=30)
pacf_values = st.pacf(cont_recovered[:,1], nlags = 30)
acf_plot = go.Figure(data=go.Bar(x=np.arange(len(acf_values)), y = acf_values))
acf_plot.add_trace(go.Bar(x=np.arange(len(pacf_values)), y = pacf_values))

acf_plot.update_layout(
    title="Recovered Autocorrelation and Partial Autocorrelation Function (ACF/PACF)",
    xaxis_title="Lags",
    yaxis_title="ACF/PACF",
    showlegend=False
)

acf_plot.show()
"""

