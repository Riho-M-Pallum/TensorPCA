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


""" 
The way that I carry out prediction is that I generate a model of say size (N,T), I want to predict new entries in the T direction,
 so for finding the loadings I only use floor( T*x ), x in (0,1) observations. With the final 1-T*x observations serving as the set that I wish
 to predict over. Prediction requires two steps. First simulating the path of the factors using an estimated AR model, then plugging those into the 
 equation for y at a future date.	
 Naturally if y_{t} = Lambda1 * F_{t} + Lambda_2 * F_{t-1} + e_{t}
 Then E[y_{t+1}] = Lambda1 E[F_{t+1}] + Lambda_2 F_{t} + E[e_{t+1}] = Lambda1 AF_{t} + Lambda2 F_{t}

 In the same way 
 E[y_{t+h}] = Lambda1E[f_{t+h}] + lambda2 E[F_{t+h-1}] = Lambda1 A^{h}f_{t} + Lambda2 A^{h-1}f_{t}?

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
# Ehh, gotta account for the sigma
# Starting off with one period forecasts

seed = np.random.randint(0,10000)# 990 # 609, 36, 990, 8550,2221# 
print(f"seed = {seed}")
Lags = 1
ar_coeffs = np.array([[0.8]])
# Ok so, for recostruction the scale components are always positive. While the true rotation is sometimes one positive ad none negative
# e.g. true is [1,-1], I recover [1,1]
n_components = 1
I = 5
T = 5 + Lags 

Y, s, M = DGP_2d_dyn_fac(I,T, n_components, Lags, ar_coeffs, noise_scale = 0, seed = seed)
Z = tPCA(Y)
s_hat, M_hat = Z.t_pca(n_components*(Lags+1))

F = M[1]
Lambda = M[0]

F_hat = M_hat["1"]
Lambda_hat = M_hat["0"]

#print(f"This is the true Y\n{Y.T}")
print(f"This is th true s {s}")
print(f"This is the estimated s {s_hat}")
print(f"True Lambda \n{Lambda/np.linalg.norm(Lambda, axis = 0 )}\n estimated Lambda \n{Lambda_hat}")
a = np.array([Lambda_hat[[0,1],0]*F_hat[0,0], Lambda_hat[[0,1],1]*F_hat[0,1]]).T
rot = np.linalg.lstsq(a, Y[[0,1],0], rcond = None)[0]
print(f"This is the rotation {rot}")
est_rec = rot[0]*np.outer(Lambda_hat[:,0], F_hat[:,0]) + rot[1]*np.outer(Lambda_hat[:,1], F_hat[:,1])
print(f"Lambda error {aux.get_cos_errors(Lambda, Lambda_hat)}")
print(f"F error {aux.get_cos_errors(F, F_hat)}")
print(f"L21 error between Y and estimate {aux.get_L21_errors(Y,est_rec)}")
#print(f"Y[0,0] {Y[[0,1],0]}, reconstructed Y[0,0] {est_rec[[0,1],0]} and the sum {s_hat['0'][0]*Lambda_hat[[0,1],0]*F_hat[0,0] + s_hat['0'][1]*Lambda_hat[[0,1],1]*F_hat[0,1]}")
# a * Lambda_hat[0,0]*F_hat[0,0] + b*Lambda_hat[0,1]*F_hat[0,1] = Y[0,0]


obs_Y = Y[:,:-1]

non_obs_Y = Y[:,-1]

Z = tPCA(obs_Y)
s_hat, M_hat = Z.t_pca(n_components*(Lags+1))

F_hat_obs = M_hat["1"]
Lambda_hat_obs = M_hat["0"]

print(f"np.shape(obs_Y){np.shape(obs_Y)}")
recovered,S, rot_matrix = aux.recover_lags(F_hat_obs, lags = Lags, dynamic_factors = n_components)
print(f"Recovered \n{recovered[-5:,:]}")
model = sm.tsa.AutoReg(recovered[:,0], lags=1)  # AR(1) model
result = model.fit()
print(f"AR params {np.array(result.params)}")
prediction = np.array([1,recovered[-1,0]])@np.array(result.params).T
prediction = [prediction, recovered[-1,0]]
print(f"Prediction {prediction}")
recovered_w_prediction = np.vstack([recovered, prediction])
print(f"recovered_w_prediction\n{recovered_w_prediction[-5:,:]}")
print(np.linalg.norm(recovered, axis = 0))
rot_inv = np.linalg.inv(rot_matrix.T)
print(f"shap rot inv {np.shape(rot_inv)} , shape recovered_w_prediction {np.shape(recovered_w_prediction)}")
print(f"Going back from recovered to F_hat, we have F_hat is\n{F_hat_obs}\nRotated back is \n{recovered_w_prediction@rot_inv}")
rotated_back_F = recovered_w_prediction@rot_inv

a = np.array([Lambda_hat_obs[[0,1],0]*rotated_back_F[0,0], Lambda_hat_obs[[0,1],1]*rotated_back_F[0,1]]).T
print(f"This is a {a}")
rot = np.linalg.lstsq(a, obs_Y[[0,1],0], rcond = None)[0]
print(f"This is the rotation to get the values to match as closely as possible {rot} and this is s_hat {s_hat}")

est_rec = rot[0]*np.outer(Lambda_hat_obs[:,0], rotated_back_F[:,0]) + rot[1]*np.outer(Lambda_hat_obs[:,1], rotated_back_F[:,1])

print(f"This is the observed Y\n{Y}\nThis is the recreated Y\n{est_rec}")

# YESSSS THIS is correct. Note the prediction accuracy depends on the amount of noise in the F process. If there is no noise and F_t = AF_{t-1}, then 
# we get perfect prediction.
