import numpy as np
from scipy.optimize import minimize
from abc import abstractmethod



def adfTest(series):
    """
    Tests for stationarity/unit root in time series
    Creates an autoregressive model of lag p (chosen through information criterion) of form:
    Δyt = α + βt + γyt-1 + δ1Δyt-1 + ... + δpΔyt-p + εt
    
    :param series: Arraylike of the series to test for stationarity
    """
    


def aic(llk, nmodel_params) -> float:
    """
    Calculates the Akaike Information Criterion for the given model
    """

    return -2.0 * llk + 2.0 * nmodel_params


class MLEModel:
    def __init__(self, endog, exog=None):
        self.endog = endog
        self.exog = exog
        self.filter = KalmanFilter(self, endog, exog)

    def fit(self, initial_params=None, method=None, func=None):
        """
        Fits the given model
        https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        """
        
        if initial_params is not None:
            x0 = initial_params
        else:
            x0 = self._init_params()
        
        params = minimize(self.fit_func, x0, args=(func), method=method)
        self.change_spec(params)

        result = Results(ssm=self.filter)
        return result
    
    
    
    def fit_func(self, params, func=None):
        """
        Sets model parameters according to change_spec, then calculates the score according to t
        """

        self.change_spec(params)

        if func is not None: # TODO implement more fitting methods
            pass
        else:
            obj_func = self.filter.nloglike
        
        return obj_func()
    

    @abstractmethod
    def _init_params(self):
        """
        Iniitalizes parameters for minimize
        Needs to be overridden
        """
        pass

    @abstractmethod
    def change_spec(self, params):
        pass


    


class KalmanFilter:
    """
    https://www.statsmodels.org/stable/statespace.html#custom-state-space-models
    Represents a Kalman Filter with a given state-space representation.
    Matrices of the SSM named thusly:

    y_t = Z_t*a_t + d_t + eps_t
    a_t+1 = T_t * a_t + c_t + R_t * eta_t

    eps_t ~ N(0, H_t)
    eta_t ~ N(0, Q_t)


    NOT YET IMPLEMENTING EXOG OR SMOOTHING
    """

    SSM_REPR_NAMES = ['Z', 'T', 'd', 'c', 'R', 'H', 'Q', 'init_state', 'init_cov']


    def __init__(self, endog, k_states, exog=None, k_posdef=None, time_invariant=True):
        self.endog = endog # (n_features, nobs)
        self.exog = exog # (n_obs, m_features)
        self.k_states = k_states
        self.time_invariant = time_invariant

        k_endog = self.endog.shape[1]
        nobs = self.endog.shape[0]
        if k_posdef is None:
            k_posdef = k_states
        elif k_posdef > k_states:
            raise ValueError("Cov matrix of process noise should have dim less than or equal to k_states")
        

        # Should be rank 3 tensor? this fine for time invariant
        self.shapes = { #TODO check this again
            'Z': (k_endog, k_states, nobs),
            'T': (k_states, k_states, nobs),
            'd': (k_endog, nobs), 
            'c': (k_states, nobs), 
            'R': (k_states, k_posdef, nobs),
            'H': (k_endog, k_endog, nobs),
            'Q': (k_posdef, k_posdef, nobs),
            'init_state': (k_states, 1),
            'init_cov': (k_states, k_states)
        }

    
    
    def setRepr(self, matrices):
        """
        Updates the transition matrices
        ssm should be a dictionary according to the statsmodel naming keys
        """

        for key in matrices:
            if not key in self.SSM_REPR_NAMES:
                raise NameError(f'{key} is not a valid key')
            elif self.time_invariant and not \
                (len(matrices[key].shape) == 2 and matrices[key].shape[:2] == self.shapes[key][:2]):
                raise ValueError(f'{key} matrix is not the correct shape, expected {self.shapes[key][:2]}, but got {matrices[key].shape[:2]}')

            if matrices[key].shape != self.shapes[key]:
                raise ValueError(f'{key} matrix is not the correct shape, expected {self.shapes[key]}, but got {matrices[key].shape}')
            
            setattr(self, "_" + key, matrices[key])
            
        

    def loglike(self):
        """
        Calculates loglike for the given state-space model
        """
        if not hasattr(self, "ssm"):
            raise KalmanFilter.StateNotSetError() # TODO move this to nloglikeobs

        self.loglikelihood = self.loglikeobs().sum(0)
        return self.loglikelihood
    
    def nloglike(self):
        return -1 * self.loglike()
    

    # TODO implement
    def loglikeobs(self):
        """
        Calculates the loglikelihood at each observation
        """
        pass






    class StateNotSetError(Exception):
        def __init__(self, message="SSM matrices must be set prior to fitting"):
            self.message = message
            super().__init__(self.message)

        def __str__(self):
            return self.message


class Results:
    def __init__(self):
        pass