import numpy as np
from scipy.linalg import null_space
from statsmodels.tsa.api import VAR

def generate_uniform_sum_to_s(S):
    # Generate n-1 random points in the range [0, S]
    cut_points = np.sort(np.random.uniform(0, S, S-1))
    
    # Add 0 at the start and S at the end to complete the interval
    cut_points = np.concatenate(([0], cut_points, [S]))
    
    # Calculate the differences between consecutive points
    random_vars = np.diff(cut_points)
    
    return random_vars


def matrix_L21_norm(A):
    """
    Compute the L2,1 norm of a matrix A.
    
    Parameters:
    A (numpy.ndarray): Input matrix of shape (m, n).
    
    Returns:
    float: The L2,1 norm of the matrix.
    """
    # Compute the L2 norm (Euclidean norm) of each column
    col_norms = np.linalg.norm(A, axis = 0)
    
    # Sum the L2 norms of all rows to get the L2,1 norm
    L21_norm = np.sum(col_norms)
    
    return L21_norm


def align_matrices_by_columns(A, B):
    """
    Ensure that corresponding columns in matrices A and B are aligned in the same direction.
    
    Parameters:
    A, B (numpy.ndarray): Input matrices of the same shape.
    
    Returns:
    numpy.ndarray: The aligned matrix B.
    """
    for j in range(A.shape[1]):
        # Compute the dot product of the corresponding columns
        dot_product = np.dot(A[:, j], B[:, j])
        
        # If the dot product is negative, flip the sign of the column in B
        if dot_product < 0:
            B[:, j] = -B[:, j]
    
    return B

def get_L21_errors(true, estimated):
    if np.shape(true)[1] > 1:
        corr_matrix = np.corrcoef(estimated.T, true.T)[:estimated.shape[1], estimated.shape[1]:]

        factor_match = np.argmax(np.abs(corr_matrix), axis=1)
        
        estimated = estimated[:, factor_match]
        
    estimated = align_matrices_by_columns(true, estimated)
        
    return matrix_L21_norm(true-estimated)


def get_L21_errors_corrs(Lambda, Mu, F, Lambda_hat, Mu_hat, F_hat):
    """
    """
    # First we determine which estimator goes with which underlying value
    if np.shape(Mu_hat)[1] > 1:
        corr_matrix = np.corrcoef(Mu_hat.T, Mu.T)[:Mu_hat.shape[1], Mu_hat.shape[1]:]

        factor_match = np.argmax(np.abs(corr_matrix), axis=1)
        if factor_match[0] == factor_match[1]:
            corr_matrix = np.corrcoef(Lambda_hat.T, Lambda.T)[:Lambda_hat.shape[1], Lambda_hat.shape[1]:]
            factor_match = np.argmax(np.abs(corr_matrix), axis=1)
    
        Lambda_hat = Lambda_hat[:, factor_match]
        Mu_hat = Mu_hat[:, factor_match]
        F_hat = F_hat[:, factor_match]
    
    Lambda_hat = align_matrices_by_columns(Lambda, Lambda_hat)
    Mu_hat = align_matrices_by_columns(Mu, Mu_hat)
    F_hat = align_matrices_by_columns(F, F_hat)
    #Now we calculate the L2 losses
    errors = dict()
    corrs = dict()
    

    #errors["Lambda"] = matrix_L21_norm(Lambda - Lambda_hat)
    #errors["Mu"] = matrix_L21_norm(Mu - Mu_hat)
    #errors["F"] = matrix_L21_norm(F - F_hat)
    
    
    for dim in range(Lambda.shape[1]):
        
        errors[f"Lambda{dim}"] = np.linalg.norm(Lambda_hat[:,dim] - Lambda[:,dim])
        errors[f"Mu{dim}"] = np.linalg.norm(Mu_hat[:,dim] - Mu[:,dim])
        errors[f"F{dim}"] = np.linalg.norm(F_hat[:,dim] - F[:,dim])
        
        corrs[f"Lambda{dim}"] = np.corrcoef(Lambda_hat[:,dim], Lambda[:,dim])[0,1]
        corrs[f"Mu{dim}"] = np.corrcoef(Mu_hat[:,dim], Mu[:,dim])[0,1]
        corrs[f"F{dim}"] = np.corrcoef(F_hat[:,dim], F[:,dim])[0,1]
    
    return errors, corrs


def get_R2_errors(true_value: list, estimated_value: list) -> np.array:




    # First we check that the true data and estimated data is of the same length
    assert len(true_value) == len(estimated_value)
    #ATA = true_data.T @ true_data
    #ATA_inv = np.linalg.inv(ATA)
    #P = true_data @ ATA_inv @ true_data.T
    #projection = P@estimated_data
    
    ATA = true_value.T @ true_value
    ATA_inv = np.linalg.inv(ATA)
    P = true_value @ ATA_inv @ true_value.T

    projection = P @ estimated_value
    #print(f"This is the projection \n{projection}\n and the estimate \n{estimated_value}\n and the true value \n{true_value}\n")

    #print(np.linalg.norm(projection, axis = 0), np.linalg.norm(estimated_value, axis = 0))

     # Compute the total sum of squares (SST)
    y_mean = np.mean(estimated_value, axis=0)
    SST = np.sum((estimated_value - y_mean) ** 2, axis=0)
    # Compute the residual sum of squares (SSE)
    SSE = np.sum((estimated_value - projection) ** 2, axis=0)
    
    # Compute R-squared
    R2 = 1 - (SSE / SST)
    
    return R2



def get_cos_errors(true_data: list, estimated_data: list) -> np.array:
    assert len(true_data) == len(estimated_data)
    
    ATA = true_data.T @ true_data
    ATA_inv = np.linalg.inv(ATA)
    P = true_data @ ATA_inv @ true_data.T
    projection = P@estimated_data
    
    # Since these are to lists of vectors, we can't just do matrix multiplication on these
    # we want to take the element wise products and then sum along the columns to get the dot products. 
    # Although we could technically do matrix multiplication and take the diagonal, this is just adding unnecessary calculations
    cos = np.sum(projection * estimated_data, axis = 0)/(np.linalg.norm(projection, axis = 0)*np.linalg.norm(estimated_data, axis = 0))
    return cos


def recover_lags(vectors, lags, dynamic_factors):
    """
        Note that there is rotational indeterminacy for 
    """
    
    recovered_vectors = np.zeros(np.shape(vectors)) # initialise the recovered vector
    number_of_vecs = np.shape(vectors)[1]
    augmented_matrix = np.array([])
    #A = np.array([vectors[:-lags,i] for i in range(number_of_vecs)]).T
    A = np.array([vectors[:-1, i] for i in range(number_of_vecs)]).T
    B = -1*np.array([vectors[1:,i] for i in range(number_of_vecs)]).T
    shape = np.shape(A)
        
    for lag in range(1,lags+1): 
        #A = np.array([vectors[:-lags, i] for i in range(number_of_vecs)]).T
        #print(f"This is the lag in the recover lags loop {lag}")
        #print(f"With bounds {lag}:{np.shape(vectors)[0]-(lags-lag)}")
        #print(f"With shape of vectors {np.shape(vectors[:,0])}")
        #B = -1*np.array([vectors[lags:np.shape(vectors)[0]-(lags-lag),i] for i in range(number_of_vecs)]).T
        #print(f"shape of B {np.shape(B)}")
        
        first_zeros = np.zeros(shape = (shape[0], shape[1]*(lag-1)))
        last_zeros = np.zeros(shape = (shape[0], shape[1]*(lags - lag)))
        
        temp = np.hstack((first_zeros, A, B, last_zeros))
        print(f"This is the shape of temp {np.shape(temp)}")
        augmented_matrix = np.vstack((augmented_matrix, temp)) if augmented_matrix.size else temp
    print(f"The augmented matrix size {np.shape(augmented_matrix)}")
    _, S, V = np.linalg.svd(augmented_matrix)
    
    #null_mask = (S <= 0.1)
    null_space = V[-dynamic_factors:, :] # Note these are row eigenvectors
    
    if np.abs(S[-dynamic_factors-1])/np.abs(S[-dynamic_factors]) < 2:
        #print(f"Relatively small second smallest eigenvalue {np.abs(S)}, {S[-2]}, {S[-1]}")
        pass
    recovered_vectors = np.zeros(shape = np.shape(vectors))
    rot_matrices = []
    rot_matrices = np.empty(shape = (0, number_of_vecs))
    for vec in range(dynamic_factors):
        single_rot = np.zeros(shape = (lags+1, number_of_vecs))
        for lag in range(lags+1):
            rot = null_space[vec, number_of_vecs*lag:number_of_vecs*(lag+1)]
            single_rot[lag,:] = rot
            #print(f"This is the rotation {rot} for lag{lag} and vec {vec}")
            recovered_vectors[:,vec*(lags+1) + lag] = null_space[vec, number_of_vecs*lag:number_of_vecs*(lag+1)] @ vectors.T
        rot_matrices = np.vstack((rot_matrices, single_rot))
        #rot_matrices.append(single_rot)
    return recovered_vectors, S, rot_matrices


def FHLR_2D_estimation(Y, q, s, H, M):
    # I assume that the last dimension of Y is the time dimension
    slices = [Y[:,i] for i in range(np.shape(Y)[1])]
    #print(f"These are the slices shape {np.shape(slices[0])}")

    autocovariances = np.empty((0, np.shape(Y)[0], np.shape(Y)[0])) # First the autocovariances are created with an increasing lag.
    # I.e. the 0th element is contemporaneous, the 1st element is one lag etc...
    for m in range(M+1):
        temp = np.zeros((np.shape(Y)[0],np.shape(Y)[0]))
        for t in range(m, len(slices)):
            cov = np.array([slices[t]]).T @ np.array([slices[t-m]])
            #print(f"Shape of slice cov {np.shape(cov)}")

            temp += cov
        #print(f"autocovariances shape {np.shape(autocovariances)}, addition shape {np.shape(temp)}")
        autocovariances = np.append(autocovariances, [temp/len(slices)], axis = 0)

    frequencies =  np.linspace(-H, H, 2*H+1)*np.pi/H
    weights = 1-np.abs(np.linspace(-M,M, 2*M+1))/(M+1)

    acf_matrices = []# In the frequency domain we have them ordered 
    #print(f"The shape of the first element of autocovariances{np.shape(autocovariances[-1])}")
    #print(f"The shape of autocovariances in general {np.shape(autocovariances)}")
    for frequency in range(len(frequencies)):
        temp = np.linspace(-M,M, 2*M+1)
        basis = np.exp(-1j*frequencies[frequency]*temp)
        reshaped_array = (weights[:M] * basis[:M])[:, np.newaxis, np.newaxis]
        a = np.sum(reshaped_array * autocovariances[:0:-1,:,:], axis = 0)
        reshaped_array = (weights[M:] * basis[M:])[:, np.newaxis, np.newaxis]
        b = np.sum(reshaped_array * autocovariances, axis = 0)
        acf_matrix = a + b
        #print(f"Shape of second product {np.shape(b)}")
        acf_matrices.append(acf_matrix / 2*np.pi)
    common_cov = []
    idio_cov = []
    for matrix in acf_matrices:
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        #print(f"Eigenvec shape {np.shape(eigenvecs[:,:q])}, eigenvals shape {np.shape(np.diag(eigenvals[:q]))}")
        common_cov.append(eigenvecs[:,:q] @ np.diag(eigenvals[:q]) @ eigenvecs[:,:q].T)
        idio_cov.append(eigenvecs[:,q:] @ np.diag(eigenvals[q:]) @ eigenvecs[:,q:].T)

    estimated_cov = []
    estimated_idio = []
    
    for k in range(M):
        basis = np.exp(1j*frequencies*k)

        estimated_cov.append(np.sum(common_cov*basis[:, np.newaxis, np.newaxis], axis = 0)/(2*H+1)*2*np.pi)
        estimated_idio.append(np.sum(idio_cov*basis[:, np.newaxis, np.newaxis], axis = 0)/(2*H+1)*2*np.pi)
        

    # And now finally to get the F projection:
    cont_cov = estimated_cov[0]
    cont_eigenvals, cont_eigenvecs = np.linalg.eigh(cont_cov)
    cont_eigenvecs = np.real_if_close(cont_eigenvecs, tol=.001) # Careful with this here!
    # Step 5) Gram schmidt in the inner product space of the idiosycratic, is this actually necessary?
    
    F_hat = Y.T @ cont_eigenvecs[:,:q*(s+1)]
    Lambda_hat  = np.linalg.inv(cont_eigenvecs[:,:q*(s+1)].T @ autocovariances[0] @ cont_eigenvecs[:,:q*(s+1)]) @ cont_eigenvecs[:,:q*(s+1)].T @ cont_cov
    M = [Lambda_hat, F_hat]
    return M, estimated_cov, estimated_idio



def Bai_Ng_determine_q(F, order, dynamic_factors = 0):
    """
    order (int): order of the VAR to estimate in F
    dynamic_factors (int): the number of dynamic factor in the model
    """
    model = VAR(F)
    fitted_model = model.fit(order)
    residuals = fitted_model.resid
    covariance_mat = np.cov(residuals.T)
    print(f"shape of Bai and Ng cov matrix {np.shape(covariance_mat)}")
    eigenvals, eigenvecs = np.linalg.eigh(covariance_mat)

    D1 = [np.sqrt(k**2/np.sum(eigenvals)) for k in eigenvals][::-1]
    D2 = [np.sqrt( np.sum(eigenvals[:k]**2)/ np.sum(eigenvals)) for k in range(len(eigenvals))][::-1]
    return D1, D2

def old_alternating_least_squares(target_Y, factors, convergence_criteria):
    """
        I assume that the tensor is made a outer( outer(F, Mu), Lambda)
        Currently only works if there are two factors.
        # Also big off this code in general. Readability and generalisabilty = 0
    """
    #F, Mu, Lambda = factors # M,L,F -> (5,8,10)
    outer = lambda x,y: np.multiply.outer(x,y)
    prev_loss = np.inf
    current_loss = 0
    counter = 0
    rotation = 0
    updated_values = [factors[0], factors[1], factors[2]]

    unraveled_y = target_Y.ravel()
    while counter < 1000 and np.abs((current_loss - prev_loss)) > convergence_criteria:
        #print(f"counter {counter}, rotatation {rotation}, current_loss {current_loss}, prev_loss {prev_loss}")
        prev_loss = current_loss
        counter += 1
             
        if rotation == 0:
            # In no particular order
            first_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,0]), updated_values[2][:,0])
            second_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,0]), updated_values[2][:,0])
            third_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,1]), updated_values[2][:,1])
            fourth_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,1]), updated_values[2][:,1])
        elif rotation == 1:
            first_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,0]), updated_values[2][:,0])
            second_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,1]), updated_values[2][:,0])
            third_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,0]), updated_values[2][:,1])
            fourth_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,1]), updated_values[2][:,1])
        elif rotation == 2:
            first_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,0]), updated_values[2][:,0])
            second_ten = outer(outer(updated_values[0][:,0], updated_values[1][:,0]), updated_values[2][:,1])
            third_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,1]), updated_values[2][:,0])
            fourth_ten = outer(outer(updated_values[0][:,1], updated_values[1][:,1]), updated_values[2][:,1])
        
        vectorised_tensors = np.array([first_ten.ravel(), second_ten.ravel(), third_ten.ravel(), fourth_ten.ravel()]).T
        rot = np.linalg.inv(vectorised_tensors.T@vectorised_tensors)@vectorised_tensors.T@unraveled_y
        #print(f"shape of rot {np.shape(rot)}")
        #print(f"Actual rot {rot}")
        updated_values[rotation] =  updated_values[rotation] @ rot.reshape(2,2).T
        
        #print(f"Shape of vectorised {np.shape(vectorised_tensors)}, shape of rot {np.shape(rot)}")
        current_loss = np.sum((unraveled_y - vectorised_tensors @ rot)**2)
        rotation = (rotation + 1) % 3
    print(f"Counter at the end {counter}")
    print(f"Final loss {current_loss}")
    return updated_values


def n_mode_outer_product(factors):
        """Compute the n-mode outer product of factor matrices."""
        result = factors[0]
        for factor in factors[1:]:
            result = np.multiply.outer(result, factor)
        return result

def alternating_least_squares(target_Y, factors, convergence_criteria, max_iters=1000):
    """
    Generalized Alternating Least Squares (ALS) for any number of factors.
    
    Arguments:
    - target_Y: Target tensor we are trying to approximate.
    - factors: List of factor matrices (one for each mode/dimension of the tensor).
               Each factor should be a matrix of shape (n_i, r), where n_i is the size of
               the i-th dimension of the tensor, and r is the rank of the factorization.
    - convergence_criteria: Convergence criterion based on the change in loss between iterations.
    - max_iters: Maximum number of iterations to perform.

    Returns:
    - updated_factors: List of updated factor matrices.
    """

    
    prev_loss = np.inf
    current_loss = 0
    counter = 0
    rotation = 0
    updated_factors = factors.copy()

    unraveled_y = target_Y.ravel()
    
    # Continue alternating until convergence or max iterations
    while counter < max_iters and np.abs(current_loss - prev_loss) > convergence_criteria:
        prev_loss = current_loss
        counter += 1
        other_factors = updated_factors[:rotation] + updated_factors[rotation+1:]
        ith_outer_prods = []
        n_factors = np.shape(updated_factors[rotation])[1]
        #print(f"Current rotation {updated_factors[rotation]}")
        for i in range(n_factors):
            ith_elements = [element[:,i].tolist() for element in other_factors]
            # Add the factors back into the position that they belong in
            for j in range(n_factors):
                temp = ith_elements.copy()
                temp.insert(rotation, updated_factors[rotation][:,j].tolist())
                #temp = [arr for arr in ith_elements if len(arr) > 0]
                # Calculate the next rank 1 matrix
                outer_prod = n_mode_outer_product(temp)
                ith_outer_prods.append(outer_prod)

        
        test = [tensor.ravel() for tensor in ith_outer_prods]
        vectorized_tensors = np.array([tensor.ravel() for tensor in ith_outer_prods]).T

        # Solve least squares problem to update the current factor
        a = np.linalg.inv(vectorized_tensors.T@vectorized_tensors)@vectorized_tensors.T
        rot = np.linalg.inv(vectorized_tensors.T@vectorized_tensors)@vectorized_tensors.T@unraveled_y
        #np.linalg.lstsq(vectorized_tensors, unraveled_y, rcond=None)[0]

        # Update the current factor matrix with the reshaped solution
        updated_factors[rotation] = updated_factors[rotation] @ rot.reshape(n_factors,n_factors).T

        # Normalise the updated factor to have norm 1
        #s = np.linalg.norm(updated_factors, axis = 1)

        # Compute loss
        current_loss = np.sum((unraveled_y - vectorized_tensors @ rot) ** 2)
        #print(f"Loss difference = {current_loss - prev_loss}")
        # Move to the next factor in the rotation
        rotation = (rotation + 1) % len(updated_factors)

    print(f"Converged after {counter} iterations.")
    return updated_factors

       