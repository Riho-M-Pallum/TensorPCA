import numpy as np
from sympy import Matrix
from scipy.linalg import null_space
import TensorPCA.auxiliary_functions as aux

np.random.seed(0)

v0 = np.array([0,1,2,3,3,5, 6, 7,8, 9, 10])
v1 = np.array([-1,0,1,2,3,3, 5, 6,7, 8, 9])
#v2 = np.zeros(shape = 9)
v2 = np.array([2,-1,0,1,2,3,3, 5, 6, 7, 8])

#t0 = np.random.normal(0,2,9)
#t1 = np.concatenate([np.random.normal(0,2,1),t0[:-1]])
#t2 = np.zeros(shape = 9)
#t2 = np.concatenate([np.random.normal(0,2,1),t1[:-1]])
t0 = np.array([3,1,2,0,-1,5, -2, -4,-5, -6, -7])
t1 = np.array([0,3,1,2,0,-1, 5, -2, -4, -5, -6])
t2 = np.array([2,0,3,1,2,0,-1, 5, -2, -4, -5])

f0 = 2*v0 + 2*v1 -3*v2 + t0 -2*t1  #+ np.random.normal(0,0.1, 9)#2vo + 2v1 
f1 = v0 - 3*v1 + v2 + 1.5*t0 + t1 - 4*t2 #+ np.random.normal(0,0.1, 9)# v0 -3v1
f2 = 2*v0 - 4*v1 + v2 - 2*t0 - t1 + 2*t2 #+ np.random.normal(0,0.1, 9)
f3 = -2*v0 + 5*v1 - 2*v2 + t0 +t1 + t2 #+ np.random.normal(0,0.1, 9)
f4 = -7*v0 + 3*v1 + 8*v2 -2*t0 + 6*t1 - 9*t2 #+ np.random.normal(0,0.1, 9)
f5 = 7*v0 + 1*v1 + 3*v2 - 5*t0 - 5*t1 - 3*t2 #+ np.random.normal(0,0.1, 9)

f_matrix = np.array([f0, f1, f2, f3, f4, f5]).T
v_matrix = np.array([v0, v1, v2]).T
t_matrix = np.array([t0, t1, t2]).T

print(f"F0, F1, F2 \n{f_matrix} with rank {np.linalg.matrix_rank(f_matrix)}")
print(f" True V matrix \n{v_matrix} with rank {np.linalg.matrix_rank(v_matrix)}")
print(f"True T matrix\n{t_matrix} with rank {np.linalg.matrix_rank(t_matrix)}")
print(f"For a combined rank of {np.linalg.matrix_rank(np.hstack([v_matrix,t_matrix]))}")
print(f"F projection error \n{aux.get_R2_errors(np.hstack([v_matrix, t_matrix]), f_matrix)}\n")


A = np.array([f0[:-2], f1[:-2], f2[:-2], f3[:-2], f4[:-2], f5[:-2]]).T

B = np.array([f0[1:-1], f1[1:-1], f2[1:-1], f3[1:-1], f4[1:-1], f5[1:-1]]).T

C = np.array([f0[2:], f1[2:], f2[2:], f3[2:], f4[2:], f5[2:]]).T

#D = np.array([f0[3:], f1[3:], f2[3:], f3[3:]]).T

zeros = np.zeros(shape = np.shape(C))
# Form the augmented matrix [A | -C]
augmented_matrix1 = np.hstack((A, -B, zeros)) # Ok so this is satisfies by all the vectors which are separated by one lag. I.e. v0/v1 + v1/v2 + t0/t1 + t1/t2 for a total of four
augmented_matrix2 = np.hstack((zeros, B, -C)) # This is satisfied by everything that satisfies a second lag so v0/v2 + t0/t2
print(f"This is A \n{np.vstack((np.hstack((A, -B)),np.hstack((A, zeros))  ))}")
#augmented_matrix3 = np.hstack((zeros, B, -C)) # This is satisfied by everything that satisfies a second lag so v0/v2 + t0/t2

#augmented_matrix3 = np.hstack((A, zeros, zeros, -D))

full_augmented = np.vstack((augmented_matrix1, augmented_matrix2))
print(f"full augmented shape {np.shape(full_augmented)}")
#n = null_space(full_augmented)
U, S, Vt = np.linalg.svd(full_augmented)
print(f"The singular values S \n{S}")
n = np.array([Vt[-1, :]]).T

print(f"Full nullspace \n{n}")
print(f"Shape of nullspace {np.shape(n)}")
tr_n = null_space(full_augmented)
print(f"True nullspace {tr_n} with shape \n {np.shape(tr_n)}")
rec_f0 = n[:6,0] @ f_matrix.T
rec_f1 = n[6:12,0] @ f_matrix.T
rec_f2 = n[12:18,0] @ f_matrix.T

rec_f_matrix = np.array([rec_f0,rec_f1,rec_f2]).T
print(f"The recovered rotation matrix \n{rec_f_matrix/1.48535267}")
print(f"It's projecton onto v {aux.get_R2_errors(v_matrix, rec_f_matrix)}")
print(f"It's projecton onto t {aux.get_R2_errors(t_matrix, rec_f_matrix)}")


print(f"Testing custom function to recover the dynamic factors")
recovered = aux.recover_lags(f_matrix, 2, 2)
print(f"Need on tuletatud vektorid\n{recovered/[1.51347543, 1.51347543, 1.51347543, -0.52063127, -0.52063127,-0.52063127]}")

cont_vecs = np.array([v2,t2]).T
one_lag_vecs = np.array([v1,t1]).T
two_lag_vecs = np.array([v0,t0]).T
print(cont_vecs)
print(f"Cont error {aux.get_R2_errors(cont_vecs, recovered[:, [2, 5]])}")
print(f"one lag error {aux.get_R2_errors(one_lag_vecs, recovered[:, [1, 4]])}")
print(f"two lag error {aux.get_R2_errors(two_lag_vecs, recovered[:, [0, 3]])}")
#print(np.hstack([n2,n3]))
#print(intersection_of_vector_sets(n1,n2,n3))

"""augmented_matrix1 = np.hstack((A, -B, zeros))
augmented_matrix2 = np.hstack((zeros, B, -C))
full_augmented = np.vstack((augmented_matrix1, augmented_matrix2))
n = null_space(full_augmented)

print(f"Full nullspace \n{n}")
rec_f0 = n[:3,0] @ f_matrix.T
rec_f1 = n[3:6,0] @ f_matrix.T
rec_f2 = n[6:,0] @ f_matrix.T
rec_f_matrix =np.array([rec_f0,rec_f1,rec_f2]).T
print(f"The alternative recovered rotation matrix \n{rec_f_matrix/[-2.61711961e-01, -2.61711961e-01, -2.61711961e-01]}")
"""

