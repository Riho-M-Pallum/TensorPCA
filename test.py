import numpy as np
from statsmodels.tsa.api import VAR
from TensorPCA.dgp import DGP_2d_dyn_fac
from sklearn.decomposition import PCA
import TensorPCA.auxiliary_functions as aux
from TensorPCA.dgp import gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda
from scipy.stats import random_correlation
from TensorPCA.tensorpca import TensorPCA as tPCA

seed = np.random.randint(0,10000)# 990 # 609, 36, 990, 8550,2221# 
print(f"seed = {seed}")
Lags = 1
ar_coeffs = np.array([[0.8, 0.7]])
n_components = 2
I = 10
J = 20
T = 30 + Lags


# So if we increase the time dimension too much with respect to the others then we do not recover all the factors well
# Something weird is going on, sometimes we recover the last two factors well and sometimes we recover them really badly, seems to be seed dependent.
# That is most likely going to be small sample problems.

Mu_corr = random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Sigma_corr = random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Mu_m = np.random.normal(0,1, size = n_components*(Lags+1))
Lambda_m = np.random.normal(0,1, size = n_components*(Lags+1))

Y, M,s = gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda(I,J,T, seed, ar_coefficients = ar_coeffs, noise_scale = 0.01, y_scale = 10,
        dynamic_factors = n_components, lags = Lags, mu_mean = Mu_m, mu_sigma = Mu_corr, lambda_mean = Lambda_m, lambda_sigma = Sigma_corr, case = 1)

F = M[2]
#Lambda = M[0]
print(f" Shape of F {np.shape(F)}")

Z = tPCA(Y)
s_hat, M_hat = Z.t_pca(n_components*(Lags+1))
F_hat = M_hat["2"]

print(f"s_hat {s_hat}")
rec, S, _ = aux.recover_lags(F_hat, 1, 2)


D1, D2 = aux.Bai_Ng_determine_q(F_hat, 4)
print(D1)
print(D2)
print(S)
