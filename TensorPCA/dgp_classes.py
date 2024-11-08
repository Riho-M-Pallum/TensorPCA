import numpy as np
import statsmodels.api as sm

class DataGeneratingProcess:
    def __init__(self, shape, factors):
        # Note that the factor AR coefficients determine the AR process for the factors.
        # While the lags and dynamic_factors govern how many factors to return.

        # Shape is (F, Lambda, Mu)
        self.shape = shape
        self.factors = factors
        
    def generate_data(self):
        """
        Placeholder for generating data.
        Override this method in child classes for different DGPs.
        """
        raise NotImplementedError



class JunsuPanDGP(DataGeneratingProcess):
    """
    Generates the tensor data as a strong factor model introduced in the paper.
    The first dimension is treated as temporal, and generates AR(1) process factors.

    Parameters
    ----------
    shape : tuple
        shape of the tensor.
    factors : int
        rank or number of factors.

    """
    def __init__(self, shape, factors):
        super().__init__(shape, factors)
        
    def generate_data(self):
        """

        Returns
        -------
        M : array of factors in the shape F, Lambda, Mu
        s : scale components for tensor
        """
        d = len(self.shape)    
        T = self.shape[0]
        # generating factor
        rho = 0.5
        sig_e = 0.1
        F = np.empty((T+100,self.factors))
        e = sig_e * np.random.normal(0,1,(T+100,self.factors))
        F[0,:] = e[0,:]
        for t in range(1,T+100):
            F[t,:] = F[t-1,:] * rho + e[t,:]
        F = F[100:,:]
        for r in range(R):
            F[:,r] = F[:,r] / LA.norm(F[:,r])
            
        # generating loadings
        M = [F]
        for j in range(1,d):
            M.append(ortho_group.rvs(self.shape[j])[:,0:self.factors])
        print(f"F {np.shape(M[0])} Lambda {np.shape(M[1])} Mu {np.shape(M[2])}")
        # generating scale component, singal strength
        s = np.sqrt(np.prod(self.shape)) * np.array(range(self.factors,0,-1))
        #F = M[0]
        #Lambda = M[1]
        #Mu = M[1]
        return M, s




class base_F_AR_DGP(DataGeneratingProcess):
    def __init__(self, shape, factors, lags = 0, ar_coefficients = [], ar_scale = 1):
        """
            Generate time factors which can follow an arbitrary AR process defined by ar_coefficients.
            Also generate factor lags 
        """
        super().__init__(shape, factors)
        self.lags = lags
        self.ar_coefficients = ar_coefficients
        self.ar_scale = ar_scale


    def generate_data(self, seed):
        np.random.seed(seed)
        T = self.shape[0] + self.lags
        f = np.zeros((T,self.factors))
        
        for factor in range(self.factors):
            # Statsmodels requires us to specify the AR polynomial to generate an AR process, hence 
            # to get the AR coefficients we have to add a 1 to the start and then take the negative of our specified AR coefficients
            # Could rewrite later to accept just the ar_polynomial!!!
            ar_params = -1*self.ar_coefficients[:,factor]
            ar_params = np.concatenate( ([1], ar_params) )
            ar_process = sm.tsa.arma_generate_sample(ar = ar_params, ma = [1], nsample = T, burnin = 0, scale = self.ar_scale)  # second value is the lag polynomial for the MA bit of the process
            f[:,factor] = ar_process
        
        F = []
        for r in range(self.lags+1):
            f_slice = f[r:T-self.lags+r, :] if self.lags > 0 else f
            for i in range(self.factors):
                F.append(f_slice[:,i])
            
        F = np.array(F).T
        M = [F]
        for shape in self.shape[1:]:
            M.append(np.random.normal(0,1, size = (shape, self.factors*(self.lags+1))))
        s = np.sqrt(np.prod(self.shape)) * np.array(range(self.factors*(self.lags+1),0,-1))/10
        print(f"F {np.shape(M[0])} Lambda {np.shape(M[1])} Mu {np.shape(M[2])}")
        
        return M, s

class correlated_Mu_Lambda_F_AR(DataGeneratingProcess):
    def __init__(self, shape, factors, lags = 0, ar_coefficients = [], means = [], correlations = []):
        """
            Ontop of generating time factors which can follow an arbitrary AR process defined by ar_coefficients, 
            we generate the rest of the factors as correlated
            
            Also generate factor lags 
        """
        super().__init__(shape, factors)
        self.lags = lags
        self.ar_coefficients = ar_coefficients
        self.means = means
        self.correlations = correlations
        
    def generate_data(self, seed):
        np.random.seed(seed)
        T = self.shape[0] + self.lags
        f = np.zeros((T,self.factors))
        
        for factor in range(self.factors):
            # Statsmodels requires us to specify the AR polynomial to generate an AR process, hence 
            # to get the AR coefficients we have to add a 1 to the start and then take the negative of our specified AR coefficients
            # Could rewrite later to accept just the ar_polynomial!!!
            ar_params = -1*self.ar_coefficients[:,factor]
            ar_params = np.concatenate( ([1], ar_params) )
            ar_process = sm.tsa.arma_generate_sample(ar = ar_params, ma = [1], nsample = T, burnin = 0, scale = 1)  # AR(2), second value is thee lag polynomial for the MA bit of the process
            f[:,factor] = ar_process
        
        F = []
        for r in range(self.lags+1):
            f_slice = f[r:T-lags+r, :] if self.lags > 0 else f
            for i in range(dynamic_factors):
                F.append(f_slice[:,i])
            
        F = np.array(F).T
        M = [F]
        
        for count, shape in enumerate(self.shape[1:]):
            M.append(np.random.multivariate_normal(means[count], correlations[count], size = shape))

        s = np.sqrt(np.prod(self.shape)) * np.array(range(self.factors*(self.lags+1),0,-1))/10
        #print(f"F {np.shape(M[0])} Lambda {np.shape(M[1])} Mu {np.shape(M[2])}")
        return M, s
        
    