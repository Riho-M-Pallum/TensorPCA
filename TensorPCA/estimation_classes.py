from TensorPCA.tensorpca import TensorPCA as tPCA
import TensorPCA.auxiliary_functions as aux
import numpy as np

class EstimationProcess:
    def __init__(self, factors):
        # Note that the factor AR coefficients determine the AR process for the factors.
        # While the lags and dynamic_factors govern how many factors to return.

        # Shape is (F, Lambda, Mu)
        self.factors = factors
    

    def estimate(self, Y, y):
        """Estimate model parameters based on the transformed tensor data"""
        raise NotImplementedError

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



class TensorPCA(EstimationProcess):
    def __init__(self, factors):
        """
            Generate time factors which can follow an arbitrary AR process defined by ar_coefficients.
            Also generate factor lags 
        """
        print(f"Numbr of facors in tensorPCA estimator class {factors}")
        super().__init__(factors)
        
    def estimate(self, Y):
        Z = tPCA(Y)
        s_hat, M_hat = Z.t_pca(self.factors)
        unrotated_data = [M_hat["0"], M_hat["1"], M_hat["2"]]
        F_hat = unrotated_data[0]
        Lambda_hat = unrotated_data[1]
        Mu_hat = unrotated_data[2]
        #print(f"This is the unrotated data {unrotated_data}")
        unrotated_s = s_hat
        #print(f"Fhat shape {np.shape(F_hat)}, Lambda shape {np.shape(Lambda_hat)}, Mu shape {np.shape(Mu_hat)}")
        factors = [F_hat, Lambda_hat, Mu_hat]
        rotated_data, s_hat = aux.alternating_least_squares(Y, factors,  convergence_criteria = 0.001)
        #s_hat = [1]*self.factors
        return unrotated_data, unrotated_s, rotated_data, s_hat
   

class usualPCA(EstimationProcess):
    def estimate(self, Y, y):
        pass


