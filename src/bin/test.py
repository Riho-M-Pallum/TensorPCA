
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import importlib
import regex as re
import os
import multiprocessing
from multiprocessing import Pool
import logging

import tensorly as tl
from tensorly.decomposition import parafac

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'simulate_params.yaml'))

print(os.getcwd())
from package import auxiliary_functions as aux
from package.dgp_classes import base_F_AR_DGP, JunsuPanDGP 
from package import dgp_classes
from package import tensorpca

"""
As a note to what happens with tensorPCA and everything. I had that the cosine similarity code was wrong. I was comparing
the cosine similarity of the estimation with the true, rather than the cosine between the projection of the estmation onto
the true and the estimation.

"""
print(np.linspace(3,25,10, dtype = int))
#
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(process)s %(levelname)s %(message)s',
    filename='results.log',
    filemode='a'
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

ar_coefficients = np.array([[0.5, 0.1, 0.7]])
dgp_instance = JunsuPanDGP(2, ar_coefficients)#base_F_AR_DGP(n_factors = 2, lags = 0, ar_coefficients = ar_coefficients, ar_scale = 1)



sample_size = (15,8,6)
seed = 1589#np.random.randint(0,10000) # 2024, 4258, 5356 is good 6124, 57, 5096, 2861, 1589 is bad 
np.random.seed(seed)
print(f"The seed is {seed}")
M, s = dgp_instance.generate_data(sample_size)
Y = dgp_classes.get_Y(M, s, 0) # We have modified the shape in the generate data call
print(f"The M1 svd {np.linalg.svd(M[1], compute_uv = False)}")
print(f"The M2 svd {np.linalg.svd(M[2], compute_uv = False)}")


s_hat, M_hat = aux.tpca(Y, (dgp_instance.lags+1)*dgp_instance.n_factors)
Z = tensorpca.TensorPCA(Y)
_, M_hat_alt = Z.t_pca((dgp_instance.lags+1)*dgp_instance.n_factors)


unfolding = aux.unfold(Y, 0)
print(f"F unfolding svd {np.linalg.svd(unfolding @ unfolding.T, compute_uv = False)}")
print(np.shape(unfolding))
M_hat = [M_hat[i] for i in range(len(M_hat))]
rotated, _ = aux.alternating_least_squares(Y, M_hat,
            convergence_criteria = 0.01, max_iters = 100)
s_hat_ALS = np.ones(shape = len(s))

reconstruction = dgp_classes.get_Y(rotated, s_hat_ALS, noise_scale = 0)
print(f"The reconstruction error is {np.sum((Y - reconstruction)**2)}")


    
print(f"How well the direct tPCA recovers F {aux.get_cos_errors(M[0], M_hat[0])}")
print(f"How well the direct Pan recovers F {aux.get_cos_errors(M[0], M_hat_alt[0])}")
print(f"How well the direct tPCA recovers L {aux.get_cos_errors(M[1], M_hat[1])}")
print(f"How well the direct tPCA recovers M {aux.get_cos_errors(M[2], M_hat[2])}")
print(f"How well the rotation recovers F {aux.get_cos_errors(M[0], rotated[0])}")
print(f"How well the rotation recovers L {aux.get_cos_errors(M[1], rotated[1])}")
print(f"How well the rotation recovers M {aux.get_cos_errors(M[2], rotated[2])}")



tensorly_parafac = parafac(Y, rank=2)
cp_decomp = tensorly_parafac[1]
#print(f"This is the cp decomp {cp_decomp[1]}")
print(f"How well the cp_decomp F {aux.get_cos_errors(M[0], cp_decomp[0])}")
print(f"How well the cp_decomp L {aux.get_cos_errors(cp_decomp[1], M[1])}")
print(f"How well the cp_decomp M {aux.get_cos_errors(M[2], cp_decomp[2])}")


manual_Y = first_bit = np.multiply.outer(np.multiply.outer(M[0][:,0], M[1][:,0]), M[2][:,0])
second_bit = np.multiply.outer(np.multiply.outer(M[0][:,1], M[1][:,1]), M[2][:,1])
manual_reconstruction = s[0]*first_bit + s[1]*second_bit
print(f"The manual reconstruction error is {np.sum((Y - manual_reconstruction)**2)}")
tensorly_reconstruction =  tl.cp_tensor.cp_to_tensor(tensorly_parafac)
print(f"Shap of tensorly reconstructon {np.shape(tensorly_reconstruction)}")
print(f"Tensorly reconstruction error {np.sum((Y - tensorly_reconstruction)**2)}")

