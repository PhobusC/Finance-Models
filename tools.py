import numpy as np
from scipy.optimize import minimize
from scipy.linalg import ldl, solve_triangular
from abc import abstractmethod
from functools import partial




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
            obj_func = partial(self.filter.nloglike, fit=True)
        
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

    SSM_REPR_NAMES = ['Z', 'T', 'd', 'c', 'R', 'H', 'Q']
    SSM_INIT_NAMES = ['init_state', 'init_cov']


    def __init__(self, endog, k_states, exog=None, k_posdef=None, time_invariant=True):
        self.endog = endog # (n_features, nobs)
        self.exog = exog # (n_obs, m_features)
        self.k_states = k_states
        self.k_posdef = k_posdef
        self.time_invariant = time_invariant
        self.diffuse=False

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
            'd': (k_endog, 1, nobs), # Hacky fix for d and c, might be better to make it rank 2 in future
            'c': (k_states, 1, nobs), 
            'R': (k_states, k_states, nobs),#(k_states, k_posdef, nobs),
            'H': (k_endog, k_endog, nobs),
            'Q': (k_states, k_states, nobs),#(k_posdef, k_posdef, nobs),
            'init_state': (k_states, 1),
            'init_cov': (k_states, k_states)#(k_posdef, k_posdef)
        }

    
    
    def setRepr(self, matrices):
        """
        Updates the transition matrices
        ssm should be a dictionary according to the statsmodel naming keys
        """
        
        
        for key in matrices:
            if not key in self.SSM_REPR_NAMES:
                raise NameError(f'{key} is not a valid key')
            elif self.time_invariant:
                potential_shape_len = len(self.shapes[key])-1
                if not (len(matrices[key].shape) == potential_shape_len and matrices[key].shape == self.shapes[key][:potential_shape_len]):
                    raise ValueError(f'{key} matrix is not the correct shape, expected {self.shapes[key][:potential_shape_len]}, but got {matrices[key].shape}')

            elif matrices[key].shape != self.shapes[key]:
                raise ValueError(f'{key} matrix is not the correct shape, expected {self.shapes[key]}, but got {matrices[key].shape}')
            
            setattr(self, "_" + key, matrices[key])


    def setInit(self, matrices, diffuse=False):
        self.diffuse=diffuse

        for key in matrices:
            if not key in self.SSM_INIT_NAMES:
                raise NameError(f'{key} is not a valid key')
            if matrices[key].shape != self.shapes[key]:
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
    

    
    def loglikeobs(self, fit=False):
        """
        Calculates the loglikelihood at each observation
        """
        # TODO implement check that ssm has been initialized
        
        state = self._init_state
        cov = self._init_cov
        
        loglikeobs = []
        for i in range(len(self.endog)):
            
            # Prediction
            obs = self.endog[:, i]
            try:
                if self.time_invariant:
                    estim_error = obs - self._Z@state - self._d
                    cov_error = self._Z@self._P@self._Z.T + self._H
                    cov_error = 0.5 * (cov_error + cov_error.T)  # Enforce symmetry

                    lower, d, perm = ldl(cov_error, lower=True)  # LDL form of F

                    # Calculate gain with LDL TODO CHECK THIS AGAIN
                    PZt = state@self._Z.T
                    W = solve_triangular(lower[perm], PZt.T, lower=True)
                    U = np.array([W[i]/d[i, i] for i in range(W.shape[0])])
                    gain = solve_triangular(lower[perm].T, U, lower=False).T

                    #gain = cov@self._Z.T@np.linalg.inv(cov_error)

                    state = self._T@(state + gain@estim_error) + self._c
                    
                    #Joseph form
                    post_cov = (np.eye(self.k_states)-gain@self._Z)@cov@(np.eye(self.k_states)-gain@self._Z).T + gain@self._H@gain.T
                    post_cov = 0.5 * (post_cov + post_cov.T)  # Enforce symmetry

                    cov = self._T@post_cov@self._T.T + self._R@self._Q@self._R.T

                else:
                    pass
                    



            except AttributeError as a:
                raise KalmanFilter.StateNotSetError(f"SSM matrices must be set prior to fitting: {str(a)}")
            
            except Exception as e:
                raise e


            if i == 0 and self.diffuse:
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
    
    def forecast(self, nsteps):
        start = len(self.filter.endog)
        return self.predict(start, start + nsteps - 1)
    

    
    @property
    @abstractmethod
    def params(self):
        """
        Gets model parameters
        """
        pass