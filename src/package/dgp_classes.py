import numpy as np
import statsmodels.api as sm
from scipy.stats import ortho_group



def get_Y(M, s, noise_scale):
    n_factors = len(s)
    shape = [np.shape(M[i])[0] for i in range(len(M))]
    #print(f"This is the shape {shape}")
    Y = noise_scale*np.random.normal(0, 1, shape)
    outer = lambda x,y: np.multiply.outer(x,y)
    for i in range(n_factors):
        temp_sol = outer(M[0][:,i], M[1][:,i])
        for j in range(2, len(shape)):
            temp_sol = outer(temp_sol, M[j][:,i])
        Y += s[i]*temp_sol
    return Y

class DataGeneratingProcess:
    def __init__(self, n_factors):
        # Note that the factor AR coefficients determine the AR process for the n_factors.
        # While the lags and dynamic_factors govern how many n_factors to return.

        # Shape is (F, Lambda, Mu)
        self.n_factors = n_factors
        self.M = None
        self.s = None
        self.shape = None
        
    def generate_data(self, shape):
        """
        Placeholder for generating data.
        Override this method in child classes for different DGPs.
        """
        raise NotImplementedError
  


class JunsuPanDGP(DataGeneratingProcess):
    """
    Generates the tensor data as a strong factor model introduced in the paper.
    The first dimension is treated as temporal, and generates AR(1) process n_factors.

    Parameters
    ----------
    shape : tuple
        shape of the tensor.
    n_factors : int
        rank or number of n_factors.

    """
    def __init__(self, n_factors, ar_coefficients):
        super().__init__(n_factors)
        self.lags = 0
        self.ar_coefficients = ar_coefficients

        
    def generate_data(self, shape):
        """

        Returns
        -------
        M : array of n_factors in the shape F, Lambda, Mu
        s : scale components for tensor
        """
        self.shape = shape
        d = len(shape)    
        T = shape[0]
        # generating factor
        rho = 0.5
        sig_e = 0.1
        
        f = np.zeros((T,self.n_factors))
        
        for factor in range(self.n_factors):
            # Statsmodels requires us to specify the AR polynomial to generate an AR process, hence 
            # to get the AR coefficients we have to add a 1 to the start and then take the negative of our specified AR coefficients
            # Could rewrite later to accept just the ar_polynomial!!!
            ar_params = -1*self.ar_coefficients[:,factor]
            ar_params = np.concatenate( ([1], ar_params) )
            ar_process = sm.tsa.arma_generate_sample(ar = ar_params, ma = [1], nsample = T, burnin = 0, scale = 1)  # second value is the lag polynomial for the MA bit of the process
            f[:,factor] = ar_process
        
        F = []
        for r in range(self.lags+1):
            f_slice = f[r:T-self.lags+r, :] if self.lags > 0 else f
            for i in range(self.n_factors):
                F.append(f_slice[:,i])
            
        F = np.array(F).T
        M = [F]
        """
        F = np.empty((T+100,self.n_factors))
        e = sig_e * np.random.normal(0,1,(T+100,self.n_factors))
        F[0,:] = e[0,:]
        for t in range(1,T+100):
            F[t,:] = F[t-1,:] * rho + e[t,:]
        F = F[100:,:]
        F = F/np.linalg.norm(F, axis = 0)
        # generating loadings
        M = [F]
        """
        for shape in shape[1:]:
            M.append(np.random.normal(0,1, size = (shape, self.n_factors*(self.lags+1))))
        
        """
        for j in range(1,d):
            M.append(ortho_group.rvs(shape[j])[:,0:self.n_factors])
        print(f"F {np.shape(M[0])} Lambda {np.shape(M[1])} Mu {np.shape(M[2])}")
        """
        # generating scale component, singal strength
        s = np.sqrt(np.prod(shape)) * np.array(range(self.n_factors,0,-1))
        
        #F = M[0]
        #Lambda = M[1]
        #Mu = M[1]
        #self.M = M
        #self.s = s
        return M, s




class base_F_AR_DGP(DataGeneratingProcess):
    def __init__(self, n_factors, lags = 0, ar_coefficients = [], ar_scale = 1):
        """
            Generate time n_factors which can follow an arbitrary AR process defined by ar_coefficients.
            Also generate factor lags 
        """
        super().__init__(n_factors)
        self.lags = lags
        self.ar_coefficients = np.array(ar_coefficients)
        self.ar_scale = ar_scale


    def generate_data(self, shape):
        """
            I assume that the dynamic AR factors are in the first dimension.
            Note that the returned factors are not normalised.
        """
        self.shape = shape
        T = shape[0] + self.lags
        f = np.zeros((T,self.n_factors))
        
        for factor in range(self.n_factors):
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
            for i in range(self.n_factors):
                F.append(f_slice[:,i])
            
        F = np.array(F).T
        M = [F]
        for shape in shape[1:]:
            M.append(np.random.normal(0,1, size = (shape, self.n_factors*(self.lags+1))))
        
        s = np.ones(self.n_factors*(self.lags+1))
        
        for i in range(len(M)):
            #print(f"The norm i is {np.linalg.norm(M[i], axis = 0)}")
            s = s*np.linalg.norm(M[i], axis = 0)
            M[i] = M[i]/np.linalg.norm(M[i], axis = 0)
        return M, s

class correlated_Mu_Lambda_F_AR(DataGeneratingProcess):
    def __init__(self, n_factors, lags = 0, ar_coefficients = [], means = [], correlations = [], ar_scale = 0.001):
        """
            Ontop of generating time n_factors which can follow an arbitrary AR process defined by ar_coefficients, 
            we generate the rest of the n_factors as correlated
            
            Also generate factor lags 
        """
        super().__init__(n_factors)
        self.lags = lags
        self.ar_coefficients = ar_coefficients
        self.means = means
        self.correlations = correlations
        self.ar_scale = ar_scale
        
    def generate_data(self, shape):
        
        self.shape = shape
        T = shape[0] + self.lags
        f = np.zeros((T,self.n_factors))
        
        for factor in range(self.n_factors):
            # Statsmodels requires us to specify the AR polynomial to generate an AR process, hence 
            # to get the AR coefficients we have to add a 1 to the start and then take the negative of our specified AR coefficients
            # Could rewrite later to accept just the ar_polynomial!!!
            ar_params = -1*self.ar_coefficients[:,factor]
            ar_params = np.concatenate( ([1], ar_params) )
            ar_process = sm.tsa.arma_generate_sample(ar = ar_params, ma = [1], nsample = T, burnin = 0, scale = self.ar_scale)  # AR(2), second value is thee lag polynomial for the MA bit of the process
            f[:,factor] = ar_process
        
        F = []
        for r in range(self.lags+1):
            f_slice = f[r:T-self.lags+r, :] if self.lags > 0 else f
            for i in range(self.n_factors):
                F.append(f_slice[:,i])
            
        F = np.array(F).T
        M = [F]
        
        for count, shape in enumerate(shape[1:]):
            M.append(np.random.multivariate_normal(self.means[count], self.correlations[count], size = shape))

        s = np.sqrt(np.prod(shape)) * np.array(range(self.n_factors*(self.lags+1),0,-1))
        return M, s
        
    