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
    def __init__(self, endog, k_states, k_posdef=None, exog=None):
        self.endog = endog
        self.exog = exog
        self.filter = KalmanFilter(endog, k_states, k_posdef=k_posdef, exog=exog)

    def fit(self, initial_params=None, method=None, func=None):
        """
        Fits the given model
        https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        """
        
        if initial_params is not None:
            x0 = initial_params
        else:
            x0 = self._init_params()
        
        results = minimize(self.fit_func, x0, args=(func), method=method)
        if not results.success:
            print(f'Optimization failed: {results.message}')


        self.change_spec(results.x)
        return Results(self.filter)
    
    
    
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
        self.k_posdef = k_posdef
        self.time_invariant = time_invariant

        k_endog = self.endog.shape[0]
        nobs = self.endog.shape[1]
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
            'init_cov': (k_posdef, k_posdef)
        }

    
    
    def setRepr(self, matrices):
        """
        Updates the transition matrices
        ssm should be a dictionary according to the statsmodel naming keys
        """
        
        
        for key in matrices:
            if not key in self.SSM_REPR_NAMES:
                raise NameError(f'{key} is not a valid key')
            elif key == 'init_state' or key == 'init_cov':
                pass
            elif self.time_invariant:
                potential_shape_len = len(self.shapes[key])-1
                if not (len(matrices[key].shape) == potential_shape_len and matrices[key].shape == self.shapes[key][:potential_shape_len]):
                    raise ValueError(f'{key} matrix is not the correct shape, expected {self.shapes[key][:potential_shape_len]}, but got {matrices[key].shape}')

            elif matrices[key].shape != self.shapes[key]:
                raise ValueError(f'{key} matrix is not the correct shape, expected {self.shapes[key]}, but got {matrices[key].shape}')
            
            setattr(self, "_" + key, matrices[key])
            
        

    def loglike(self):
        """
        Calculates loglike for the given state-space model
        """
        self.loglikelihood = self.loglikeobs().sum(0)
        return self.loglikelihood
    
    def nloglike(self):
        return -1 * self.loglike()
    

    
    def loglikeobs(self):
        """
        Calculates the loglikelihood at each observation
        """
        # TODO implement check that ssm has been initialized
        
        state = self._init_state
        cov = self._init_cov
        
        loglikeobs = []
        for i in range(len(self.endog)):
            
            # Prediction
            obs = self.endog[i]
            try:
                if self.time_invariant:
                    F = cov + self._Q
                    K = cov + np.linalg.inv(F) # TODO CHECK

                    estim_error = obs - self._Z@state - self._d
                    pred_state = state + self._F@self._P.T@np.linalg.inv(self._F)@estim_error

                else:
                    pass
                    



            except NameError as n:
                print(f"SSM must be fully specified prior to fitting\n{n}")
                raise n
            except Exception as e:
                raise e


            if i == 0:
                loglike = 0.0
            else:
                loglike = 0.0#TODO Implement 
            
            loglikeobs.append(loglike)
        
        
        return np.array(loglikeobs)







    # TODO consider adding a one pass filter method that adds useful stats as attributes
    class StateNotSetError(Exception):
        def __init__(self, message="SSM matrices must be set prior to fitting"):
            self.message = message
            super().__init__(self.message)

        def __str__(self):
            return self.message


class Results:
    def __init__(self, filter):
        self.filter = filter

    def predict(self, start, end):
        """
        Predicts endog from start to end
        """
        pass

    @property
    @abstractmethod
    def params(self):
        """
        Gets model parameters
        """
        pass