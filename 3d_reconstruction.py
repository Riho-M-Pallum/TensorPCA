import numpy as np

from TensorPCA.tensorpca import TensorPCA
from TensorPCA.dgp import DGP
from TensorPCA.dgp import gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda
from TensorPCA.base import factor2tensor
from TensorPCA.tensorpca import TensorPCA as tPCA
import TensorPCA.auxiliary_functions as aux






# Remember that we only recover the scale up to sign.
seed = 1#np.randint(0,10000) 
Lags = 1
ar_coeffs = np.array([[0.8]])
n_components = 1
components = n_components*(Lags+1)

R = 2 # rank
# tensor size TxNxJ
T = 20+Lags
I = 10
J = 5

Mu_corr = np.eye(n_components*(Lags+1))#random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Sigma_corr = np.eye(n_components*(Lags+1))#random_correlation.rvs(aux.generate_uniform_sum_to_s(n_components*(Lags+1)))
Mu_m = np.zeros(shape = n_components*(Lags+1))#np.random.normal(0,1, size = n_components*(Lags+1))
Lambda_m = np.zeros(shape = n_components*(Lags+1))# np.random.normal(0,1, size = n_components*(Lags+1))


Y, M, s = gen_non_orthogonal_multiple_dynamic_factors_correlated_mu_lambda(I,J,T, seed, ar_coefficients = ar_coeffs, noise_scale = 0., y_scale = 5,
		        dynamic_factors = n_components, lags = Lags, mu_mean = Mu_m, mu_sigma = Mu_corr, lambda_mean = Lambda_m, lambda_sigma = Sigma_corr, case = 1)
		
print(f"The Shape of Y {np.shape(Y)}")

F = M[2]
Lambda = M[1]
Mu = M[0]

print(f"Shape of Lambda {np.shape(Lambda)}, shape of Mu {np.shape(Mu)}, shape of F {np.shape(F)}")
reconstructed_y = factor2tensor(s, M)
print(f"Distance to true from true automatic reconstruction = {np.sum((reconstructed_y - Y)**2)}")

first_bit = np.multiply.outer(np.multiply.outer(Mu[:,0], Lambda[:,0]), F[:,0])
second_bit = np.multiply.outer(np.multiply.outer(Mu[:,1], Lambda[:,1]), F[:,1])

manual_reconstruction = s[0]*first_bit + s[1]*second_bit

print(f"Distance to true from true manual reconstruction = {np.sum((manual_reconstruction - Y)**2)}")

Z = tPCA(Y)
s_hat, M_hat = Z.t_pca(R)
print(f"S_hat {s_hat}")
print(f"{np.shape(np.array([s_hat['0']]))}, {np.shape(M_hat['0'])}")
F_hat = M_hat["2"] #* np.array([s_hat["0"]])
Lambda_hat = M_hat["1"] #* s_hat["1"]
Mu_hat = M_hat["0"] #* s_hat["2"]

print(f"Shape of Lambda {np.shape(Lambda_hat)}, shape of Mu {np.shape(Mu_hat)}, shape of F {np.shape(F_hat)}")

s_t = s_hat["1"]

M_t = [Mu_hat, Lambda_hat, F_hat]
reconstructed_y_t = factor2tensor(s_t, M_t)
print(f"f cos error {aux.get_cos_errors(F, F_hat)}")
print(f"L cos error {aux.get_cos_errors(Lambda, Lambda_hat)}")
print(f"M cos error {aux.get_cos_errors(Mu, Mu_hat)}")
print(f"distance between naive resonstruction and true {np.sum((reconstructed_y_t-Y)**2)}")

updated_values = aux.alternating_least_squares(Y, F_hat, Lambda_hat, Mu_hat, 0.00001) # M, L, F -> 5,4,3 | L,M,F -> 5,3,4 | M,F,L -> 4,5,3

tensor = factor2tensor([1,1], updated_values)


print(f"loss to updated tensor {np.sum((Y - tensor)**2)}")

print(f"Fucking beautiful!!!")

print(f"Do we actually get something close to the true values?")

#print(updated_values[2])
#print(F)
"""
# Generate a random tensor factor model and store the tensor
Y, s, M = DGP((T,N,J),R)
print(f"Shape of Y {np.shape(Y)}")
Lambda = M[2]
Mu = M[1]
F = M[0]

print(f"Shape of Lambda {np.shape(Lambda)}, shape of Mu {np.shape(Mu)}, shape of F {np.shape(F)}")
reconstructed_y = factor2tensor(s, M)

first_bit = np.multiply.outer(np.multiply.outer(F[:,0], Mu[:,0]), Lambda[:,0])
second_bit = np.multiply.outer(np.multiply.outer(F[:,1], Mu[:,1]), Lambda[:,1])

manual_reconstruction = s[0]*first_bit + s[1]*second_bit
print(f"True s {s}")
print(f"Distance to true from true manual reconstruction = {np.sum((manual_reconstruction - Y)**2)}")
#print(f"This is the first bit {first_bit}")
#print(f"This is F[0,0], {F[0,0]}, {Lambda[0,0]}, {Mu[0,0]}")
#print(f"{F[0,0] * Lambda[0,0] * Mu[0,0]}")
#mat = np.array([[first_bit[0,0,0], second_bit[0,0,0]], [first_bit[0,1,0], second_bit[0,1,0]]])
#target = np.array([[Y[0,0,0]], [Y[0,1,0]]])
#rot = np.linalg.lstsq(mat, target, rcond = None)[0]
#print(f"Rot is {rot}, s is {s}")
#print(f"This is the rot {s[0]*first_bit + s[1]*second_bit}")
#print(Y)
#print(f"{F[0,0] * Lambda[0,0] * Mu[0,0] * s[0]}, ")
#print(f"frst bit divided by Y{Y/first_bit}")
#print(f"Mu[0] = {Y[0,0,0]/(Lambda[0,0] * F[0,0] * s[0])} = {Mu[0,0]} = {Y[0,0,1]/(Lambda[0,0] * F[1,0] * s[0])}")

Z = tPCA(Y)
s_hat, M_hat = Z.t_pca(R)
print(f"S_hat {s_hat}")
print(f"{np.shape(np.array([s_hat['0']]))}, {np.shape(M_hat['0'])}")
Lambda_hat = M_hat["2"] #* np.array([s_hat["0"]])
Mu_hat = M_hat["1"] #* s_hat["1"]
F_hat = M_hat["0"] #* s_hat["2"]

s_t = s_hat["1"]
M_t = [F_hat, Mu_hat, Lambda_hat]
reconstructed_y_t = factor2tensor(s_t, M_t)
print(f"s_t {s_t}")
print(f"M_t {M_t}")
print(f"f cos error {aux.get_cos_errors(F, F_hat)}")
print(f"L cos error {aux.get_cos_errors(Lambda, Lambda_hat)}")
print(f"M cos error {aux.get_cos_errors(Mu, Mu_hat)}")

print(f"distance between naive resonstruction and true {np.sum((reconstructed_y_t-Y)**2)}")

Lambda_rot = np.linalg.lstsq(Lambda, Lambda_hat, rcond = None)[0]
Mu_rot = np.linalg.lstsq(Mu, Mu_hat, rcond = None)[0]
F_rot = np.linalg.lstsq(F, F_hat, rcond = None)[0].T
print(f"Lambda rot \n{Lambda_rot}, mu rot \n{Mu_rot}, F rot \n{F_rot}")

print(f"F hat \n{F_hat}, F \n{F_rot.T @ F.T}")

alpha = F_rot[0,0]
beta = F_rot[0,1]
gamma = F_rot[1,0]
delta = F_rot[1,1]

rec = factor2tensor([1,1], M)
print(f"The reconstructed_y_t {rec}")
temp_Mu = Mu #*s
outer = lambda x,y: np.multiply.outer(x,y)
thingy = alpha * outer(outer(F_hat[:,0], temp_Mu[:,0]), Lambda[:,0]) +\
		 delta * outer(outer(F_hat[:,1], temp_Mu[:,1]), Lambda[:,1]) +\
		 beta * outer(outer(F_hat[:,1], temp_Mu[:,0]), Lambda[:,0]) +\
		 gamma * outer(outer(F_hat[:,0], temp_Mu[:,1]), Lambda[:,1])

print(f"And the thingy {thingy}")


# Now let's try and solve the lstsq problem
first_ten = outer(outer(F_hat[:,0], Mu[:,0]), Lambda[:,0])
second_ten = outer(outer(F_hat[:,1], Mu[:,0]), Lambda[:,0])
third_ten = outer(outer(F_hat[:,0], Mu[:,1]), Lambda[:,1])
fourth_ten = outer(outer(F_hat[:,1], Mu[:,1]), Lambda[:,1])
#print(f"First unraveled tensor {np.shape(first_ten)}")
unraveled_y = Y.ravel()
vectorised_tensors = np.array([first_ten.ravel(), second_ten.ravel(), third_ten.ravel(), fourth_ten.ravel()]).T
print(f"vectorised_tensors shape {np.shape(vectorised_tensors)}")
rot = np.linalg.inv(vectorised_tensors.T@vectorised_tensors)@vectorised_tensors.T@unraveled_y

print(f"And the correct rotation is {rot}")
print(f"F hat shape {np.shape(F_hat)}")
print(f"reshaped rotation = {rot.reshape(2,2)}")
print(f"rot times F {F_hat @ rot.reshape(2,2).T}")
print(f"The distance to Y is {np.sum((Y - rot[0]*first_ten - rot[1]*second_ten - rot[2]*third_ten - rot[3]*fourth_ten)**2)}")
print(f"Bloody wonderful!")
F1 = rot[0] * F_hat[:,0] + rot[1] * F_hat[:,1]
F2 = rot[2] * F_hat[:,0] + rot[3] * F_hat[:,1]
print(f"F1 {F1}, and F2 \n {F2}")

#first_ten = outer(outer(rot[0]*F_hat[:,0], Mu[:,0]), Lambda[:,0])
#second_ten = outer(outer(rot[1]*F_hat[:,1], Mu[:,1]), Lambda[:,1])
#third_ten = outer(outer(rot[2]*F_hat[:,1], Mu[:,0]), Lambda[:,0])
#fourth_ten = outer(outer(rot[3]*F_hat[:,0], Mu[:,1]), Lambda[:,1])

#print(f"So the updated one should be {rot[0]*F_hat[:,0]+rot[2]*F_hat[:,1]}")
first_bit = outer(outer(F1, Mu[:,0]), Lambda[:,0])
second_bit = outer(outer(F2, Mu[:,1]), Lambda[:,1])
print(f"loss to manual updated tensor {np.sum((Y - first_bit - second_bit)**2)}")
#print(f"are they the same \n{np.sum(first_ten + third_ten - outer(outer(F1, Mu[:,0]), Lambda[:,0]))}")
#print(f"The new shape {np.shape(outer(outer(F1, Mu[:,0]), Lambda[:,0]))}")
updated_values = aux.alternating_least_squares(Y, Lambda, Mu, F_hat, 0.001)
print(f"updated_values {updated_values}")

tensor = factor2tensor([1,1], updated_values)

first_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,0]), updated_values[2][:,0])
#fourth_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,1]), updated_values[2][:,1])
second_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,1]), updated_values[2][:,1])
#third_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,0]), updated_values[2][:,0])

print(f"loss to updated tensor {np.sum((Y - tensor)**2)}")

print(f"Fucking beautiful!!!")












aa
first_bit = np.multiply.outer(np.multiply.outer(Lambda_hat[:,0], Mu_hat[:,0]), F_hat[:,0])
first_bit_of_first_slice = np.multiply.outer(Lambda_hat[:,0], F_hat[:,0]) 
second_bit_of_first_slice = np.multiply.outer(Lambda_hat[:,1], F_hat[:,1]) 

print(f"Lambda_hat {Lambda_hat}, F_hat {F_hat}")
print(f"First bit of first slice{first_bit_of_first_slice}")
first_slice_of_Y = Y[:,0,:]
second_slice_of_Y = Y[:,1,:]

mat1 = first_bit_of_first_slice[:2,0]
mat2 = second_bit_of_first_slice[:2,0]
rot = np.linalg.lstsq(np.vstack((mat1, mat2)).T, first_slice_of_Y[:2,0], rcond = None)[0]
print(mat1)
print(f"first_bit_of_first_slice {np.vstack((mat1, mat2))}, first slice of Y {first_slice_of_Y[:2,0]}, rot {rot}")
print(f"rot div by s_hat = {rot/s_hat['1']}")
print(f"Mu_hat {Mu_hat}")
print(f"the first slice of Y {Y[:,0,:]}")
print(f"The first column of the first slice shoule be {rot[0]*F_hat[0,0]*Lambda_hat[:,0] + rot[1]*F_hat[0,1]*Lambda_hat[:,1]}")
print(f"The second column should be {rot[0]*F_hat[1,0]*Lambda_hat[:,0] + rot[1]*F_hat[1,1]*Lambda_hat[:,1]}")
print(f"Recovered first slice {rot[0]*first_bit_of_first_slice + rot[1]*second_bit_of_first_slice}")

first_bit_of_second_col = F_hat[1,0]*Lambda_hat[:,0]
second_bit_of_second_col = F_hat[1,1]*Lambda_hat[:,1]

mat = np.vstack((first_bit_of_second_col, second_bit_of_second_col))
print(f"first_bit_of_second_col {first_bit_of_second_col},mat {mat.T}")
second_col_of_first_slice = first_slice_of_Y[:,1]
rot = np.linalg.lstsq(mat.T, second_col_of_first_slice, rcond = None)[0]
print(f"the target {first_slice_of_Y}")
print(f"And with the entire bit {first_bit_of_first_slice*rot[0] +second_bit_of_first_slice*rot[1]}")





print(f"lambda outer F div first slice {lambda_outer_F/first_slice_of_Y}, good they are the same")
print(f"Lambda outer f div second slice {lambda_outer_F/second_slice_of_Y}, good it is the same")
alt_mu_hat =  np.array([[1/(lambda_outer_F/first_slice_of_Y)[0,0]], [1/(lambda_outer_F/second_slice_of_Y)[0,0]]])/s_hat['0']
print(f"true Mu {Mu}, \nMu_hat {Mu_hat}, \nalternative Mu hat {alt_mu_hat}")
print(f"alternative mu hat in span of Mu hat? {aux.get_cos_errors(Mu_hat, alt_mu_hat)}")
print(f"In span of original Mu? {aux.get_cos_errors(Mu, alt_mu_hat)}")
print(f"Mu hat / Mu {Mu_hat/Mu}")
print(f"first_bit div true {Y/(first_bit*s_hat['0'])}")

print(f"Error between estimated reconstruction and true {np.sum(-1*first_bit*s_hat['0'] - Y)}")


second_bit = np.multiply.outer(np.multiply.outer(Lambda_hat[:,1], Mu_hat[:,1]), F_hat[:,1])

mat = np.array([[first_bit[0,0,0], second_bit[0,0,0]], [first_bit[0,1,0], second_bit[0,1,0]]])
target = np.array([[Y[0,0,0]], [Y[0,1,0]]])
rot = np.linalg.lstsq(mat, target, rcond = None)[0]

print(f"Rot is {rot}, s is {s}, is shat {s_hat}")
#print(f"This is the L21 error {aux.get_L21_errors(Y, rot[0]*first_bit + rot[1]*second_bit)}")
print(f"square sum of residuals {np.sum((Y-rot[0]*first_bit + rot[1]*second_bit)**2)}")
print(f"This is the rot {rot[0]*first_bit + rot[1]*second_bit}")
print(Y)

# OK so now I get the first slice correct, but not the other two, or at least not nearly as well as the first one. What the fuck???

"""