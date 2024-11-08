import numpy as np
from sympy import Matrix
from scipy.linalg import null_space
import TensorPCA.auxiliary_functions as aux
from sklearn.decomposition import PCA

np.random.seed(0)

L = np.array([np.array([2,3,4]), np.array([5,6,7])]).T
#L1 = np.array([[-2,0,2], [1,5,2]]).T
F = np.array([[0,-1,1,2], np.array([1,0,-1,1])]).T
#F1 = np.array([[-3,2,1,2], [0, -3, 2, 1]]).T

scales = np.array([[1],[1]])
data = 0
first = scales[0]*np.multiply.outer(L[:,0], F[:,0])
second = scales[1]*np.multiply.outer(L[:,1], F[:,1])
print(f"first {first}")
print(f"second {second}")
print(f"First plus second {first + second}")
data = L @ F.T
print(f"L@F.T \n{data}")

pca = PCA(n_components=2)  # Specify number of principal components to keep
U,S,Vh = np.linalg.svd(data)

left_svd = U[:,[0,1]]
right_svd = Vh.T[:,[0,1]]
singular_values = S[:2]
print(U)
print(f"S {S}")
print(f"Vh {Vh}")

reconstructed = (left_svd *singular_values) @ right_svd.T 

print(f"reconstructed after SVD \n{reconstructed}")
print(f"Reconstruction error {np.sum((reconstructed-data)**2)}")

print(f"F cos error {aux.get_cos_errors(F, right_svd)}")
print(f"L cos error {aux.get_cos_errors(L, left_svd)}")
left_svd_scaled = left_svd *singular_values
print(f"Left svd scaled {left_svd_scaled}")
lambda_rotation = np.linalg.lstsq(L, left_svd_scaled)[0]
print(f"lambda_rotation{lambda_rotation}")
F_rotation = np.linalg.lstsq(F, right_svd)[0]
print(f"F rotation {F_rotation}")

a = lambda_rotation[0,0]
b = lambda_rotation[1,0]
c = lambda_rotation[0,1]
d = lambda_rotation[1,1]

alpha = F_rotation[0,0]
beta = F_rotation[1,0]
gamma = F_rotation[0,1]
delta = F_rotation[1,1]

print(f"Lambda shape {np.shape(F[:,0].T)}")
outer = lambda x,y : np.multiply.outer(x,y)

reconstructed = (alpha*a + c*gamma)*outer(L[:,0],F[:,0]) + (beta*b + delta*d)*outer(L[:,1], F[:,1]) +\
				(a*beta + c*delta)*outer(L[:,0], F[:,1]) + (b*alpha + gamma*d)*outer(L[:,1], F[:,0])
print(reconstructed)
#print(outer(L[:,0],F[:,0]) + outer(L[:,1],F[:,1]))
print(a*L[:,0] + b*L[:,1])
print(left_svd_scaled)


print(f"What are the coefficients like? alpha a + c gamma {b*alpha}. {gamma*d}")