import numpy as np
import math
from scipy.optimize import minimize
from scipy.linalg import ldl, solve_triangular
from abc import abstractmethod
from functools import partial


def ols(y, X):
    """
    Returns Beta hat matrix under OLS assumptions
    Y is data
    X is regressors
    """

    # Since X.T@X is symmetric, maybe use LDL decomposition
    return np.linalg.inv(X.T@X)@X.T@y

def loglike_ols(obs, regressors, params, var=None):
    """
    Calculates loglikelihood given parameters
    Params: np array representing model parameters
    var: float or None representing the variance of the data
        if var=None, calculate Average Squared Residual
    """
    nobs = len(obs)
    SSR = np.sum((obs - regressors@params)**2)

    if var is None:
        var = SSR/nobs

    return (-nobs/2)*(math.log(math.pi) + math.log(var) + 1)

# There's also an ADF-GLS test?
def adfTest(series, criterion="aic", model=1, conf_level=0.05):
    """
    Tests for stationarity/unit root in time series
    Creates an autoregressive model of lag p (chosen through information criterion) of form:
    Δyt = α + βt + γyt-1 + δ1Δyt-1 + ... + δpΔyt-p + εt
    
    :param series: Arraylike of the series to test for stationarity

    param model: int signifying which regression to fit
        model = 1: no constant, no drift
                2: constant only
                3: constant and drift
    """

    series = series.squeeze()

    if len(series.shape) >= 2:
        raise ValueError("Series should be 1D")

    # Fit OLS model to differenced series, using information criterion
    # Best_model is a tuple, (max lag, criterion score)
    best_model = None

    # Schwert/Ng-Perron rule?
    diff_series = np.diff(series)
    diff_len = len(diff_series)
    

    

    if model not in [1, 2, 3]:
        raise ValueError("Param model should be from values [1, 2, 3]")
    
    # Matrices for OLS
    # Matrices are in ascending order
    p_max = math.floor(12 * math.pow(len(series)/10, 0.25))  # Schwert rule
    for p in range(1, p_max+1):
        if model == 1: # no constant no drift
            
            X = np.empty(shape=(diff_len-p, p+1), dtype=float)
            X[:, 0] = series[p:-1]
            for i in range(1, p+1):
                X[:, i] = diff_series[p-i: diff_len-i]

            y = np.expand_dims(diff_series[p:], axis=-1)


        elif model == 2: # constant only
            X = np.empty(shape=(diff_len-p, p+2), dtype=float)
            X[:, 0] = series[p:-1]
            X[:, 1] = np.ones(shape=(diff_len-p))
            for i in range(1, p+1):
                X[:, i+1] = diff_series[p-i: diff_len-i]
            
            y = np.expand_dims(diff_series[p:], axis=-1)

        elif model == 3:
            X = np.empty(shape=(diff_len-p, p+3))
            X[:, 0] = series[p:-1]
            X[:, 1] = np.ones(shape=(diff_len-p))
            X[:, 2] = np.arange(start=p+2, stop=diff_len+2)
            for i in range(1, p+1):
                X[:, i+2] = diff_series[p-i: diff_len-i]

            y = np.expand_dims(diff_series[p:], axis=-1)

        
        beta_hat = ols(y, X)

        # Calculate Loglik and AIC
        if best_model is None:
            best_model = (p, aic(loglike_ols(y, X, beta_hat), beta_hat.shape[0]))
        else:
            model_aic = aic(loglike_ols(y, X, beta_hat), beta_hat.shape[0])
            if model_aic < best_model[1]:
                best_model = (p, model_aic)



    gamma = best_model[0, 0]
    # Calculate critical value


    return best_model

def adfTable():
    """
    I might try to calculate the values to practice Monte Carlo Simulation
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

        # Handle failed optimization?
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
    FILTER_TRACK = ['loglike', 'pred', 'state', 'cov', 'innov', 'innov_var', 'gain'] # check if errors should be tracked


    def __init__(self, endog, k_states, exog=None, k_posdef=None, time_invariant=True):
        self.endog = endog # (n_features, nobs)
        self.exog = exog # (n_obs, m_features)
        self.k_states = k_states
        self.k_posdef = k_posdef
        self.time_invariant = time_invariant
        self.diffuse=False

        k_endog = self.endog.shape[0]
        self.k_endog = k_endog

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

        return np.array(self.filter(returns=['loglike'], fit=fit)['loglike'])
    
    def filter(self, start=0, stop=None, returns=['loglike', 'state', 'cov', 'innov', 'innov_var', 'gain'], fit=False):
        """
        Returns should be a list of possible things to track, as specificed by FILTER_TRACK
        Fit should only be used through loglikeobs and fit_func
        Goes from start (inclusive) to stop (inclusive)
        """
        if stop is None:
            stop = self.endog.shape[1] - 1


        # Checks for start and stop here
        if start < 0 or start > self.endog.shape[1]:
            raise ValueError("Start must be at least 0 and at most nobs")
        if stop < start:
            raise ValueError("Start must be less than stop")
        


        if returns is not None:
            stats = {}
            for r in returns:
                if r in self.FILTER_TRACK:
                    stats[r] = []
                else:
                    raise NameError(f'{r} is not a valid return')

        state = self._init_state
        cov = self._init_cov
        
        m = {} # Stores the ssm matrices for each time step
        if self.time_invariant:
            for key in self.SSM_REPR_NAMES:
                m[key] = getattr(self, "_" + key)

        for t in range(start, stop + 1):
            try:

                if t < self.endog.shape[1]:
                    obs = self.endog[:, t]
                    if not self.time_invariant:
                        for key in self.SSM_REPR_NAMES:
                            m[key] = getattr(self, "_" + key)[:, :, t]
                else:
                    obs = None

                
                if obs is not None:
                    if np.isnan(obs).any():
                        empty = ~np.isnan(obs).squeeze() # Should be shape (k_endog,)
                        obs_masked = obs[empty]
                        Z_masked = m['Z'][empty, :]
                        d_masked = m['d'][empty]
                        H_masked = m['H'][np.ix_(empty, empty)]
                        elem = np.sum(empty)

                    else:
                        obs_masked = obs
                        Z_masked = m['Z']
                        H_masked = m['H']
                        d_masked = m['d']
                        elem = self.k_endog
                    

                    # Filters using estimation errors
                    pred = Z_masked@state + d_masked # d is the wrong shape if missing obs
                    innov = obs_masked - pred
                    innov_var = Z_masked@cov@Z_masked.T + H_masked
                    innov_var = 0.5 * (innov_var + innov_var.T)  # Enforce symmetry

                    lower, d, perm = ldl(innov_var, lower=True)  # LDL form of F

                    # Calculate gain with LDL
                    PZt = cov@Z_masked.T
                    W = solve_triangular(lower[perm], PZt.T, lower=True)
                    U = np.array([W[j]/d[j, j] for j in range(d.shape[0])])
                    gain = solve_triangular(lower[perm].T, U, lower=False).T

                    #gain = cov@self._Z.T@np.linalg.inv(innov_var)

                    state = m['T']@(state + gain@innov) + m['c']
                    
                    #Joseph form
                    post_cov = (np.eye(self.k_states)-gain@Z_masked)@cov@(np.eye(self.k_states)-gain@Z_masked).T + gain@H_masked@gain.T
                    post_cov = 0.5 * (post_cov + post_cov.T)  # Enforce symmetry

                    cov = m['T']@post_cov@m['T'].T + m['R']@m['Q']@m['R'].T

                    

                
                else:
                    pred = m['Z']@state + m['d']
                    state = m['T']@state + m['c']
                    cov = m['T']@cov@m['T'].T + m['R']@m['Q']@m['R'].T
                    innov = None
                    innov_var = None
                    gain = None
 
            


            except AttributeError as a:
                raise KalmanFilter.StateNotSetError(f"SSM matrices must be set prior to fitting: {str(a)}")
            
            except Exception as e:
                raise e
            

            if returns is not None:
                for r in returns:
                    if r == 'loglike':
                        if (t == 0 and self.diffuse) or obs is None:
                            loglike = 0.0
                        else:
                            innov_var_size = 1
                            for i in range(d.shape[0]): # is there a prettier way to do this?
                                innov_var_size *= d[i, i]
                            
                            # Solve for quadratic term
                            z = solve_triangular(lower[perm], innov, lower=True)
                            quad = np.sum(z.dot(np.array([z[j]/d[j, j] for j in range(d.shape[0])])))

                            loglike = -0.5*(np.log(innov_var_size) + quad)
                            if not fit:
                                loglike += -0.5*elem*math.log(math.pi * 2)
                        
                        stats[r].append(loglike)
                    elif r == 'obs':
                        # Changes shape of pred back if needed
                        if obs is not None and elem < self.k_endog:
                            pred_full = np.full((self.k_endog, 1), np.nan)
                            pred_full[empty] = pred
                            pred = pred_full
                        
                        stats[r].append(pred)

                    else: 
                        # Add thing that expands pred back to original shape if masked
                        stats[r].append(locals()[r] if locals()[r] is not None else np.nan) # Is this a good way of doing this?
        
        return stats

    class StateNotSetError(Exception):
        def __init__(self, message="SSM matrices must be set prior to fitting"):
            self.message = message
            super().__init__(self.message)

        def __str__(self):
            return self.message


class Results:
    def __init__(self, model):
        self.model = model
        self.stats = self.model.filter.filter()

    def predict(self, start, end):
        """
        Predicts endog from start to end
        """
        obs = self.model.filter.filter(start=start, stop=end, returns=['pred'])
        return obs
    
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