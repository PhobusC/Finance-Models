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
    lower, d, perm = ldl(X.T@X, lower = True)
    DLB = solve_triangular(lower[perm], X.T@y, lower=True)
    LB = np.array([DLB[i]/d[i, i] for i in range(DLB.shape[0])])
    beta_hat = solve_triangular(lower[perm].T, LB, lower=False)


    # return np.linalg.inv(X.T@X)@X.T@y
    return beta_hat

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

# Engel-Granger test?
def mackinnon_crit_values(model_type, sig, N=1):
        """
        Basically copy pasted from statsmodels, I'm not doing allat rn
        N is number of cointegrated series
        sig is significance level

        Returns the MacKinnon coefficients for the given N and sig
        https://www.econstor.eu/bitstream/10419/67744/1/616664753.pdf 
        """
        # These are the new estimates from MacKinnon 2010
        # the first axis is N - 1
        # the second axis is 1 %, 5 %, 10 %
        # the last axis is the coefficients

        if model_type not in ['nc', 'c', 'ct', 'ctt']:
            raise ValueError("Param model should be from values ['nc', 'c', 'ct', 'ctt']")

        if sig not in [0.01, 0.05, 0.10]:
            raise ValueError("Significance level should be from [0.01, 0.05, 0.10]")

        tau_nc_2010 = [[ [-2.56574,-2.2358,-3.627,0], # N = 1
                        [-1.94100,-0.2686,-3.365,31.223],
                        [-1.61682, 0.2656, -2.714, 25.364]]]
        tau_nc_2010 = np.asarray(tau_nc_2010)

        tau_c_2010 = [[ [-3.43035,-6.5393,-16.786,-79.433], # N = 1, 1%
                        [-2.86154,-2.8903,-4.234,-40.040],  # 5 %
                        [-2.56677,-1.5384,-2.809,0]],       # 10 %
                    [ [-3.89644,-10.9519,-33.527,0],      # N = 2
                        [-3.33613,-6.1101,-6.823,0],
                        [-3.04445,-4.2412,-2.720,0]],
                    [ [-4.29374,-14.4354,-33.195,47.433], # N = 3
                        [-3.74066,-8.5632,-10.852,27.982],
                        [-3.45218,-6.2143,-3.718,0]],
                    [ [-4.64332,-18.1031,-37.972,0],      # N = 4
                        [-4.09600,-11.2349,-11.175,0],
                        [-3.81020,-8.3931,-4.137,0]],
                    [ [-4.95756,-21.8883,-45.142,0],      # N = 5
                        [-4.41519,-14.0405,-12.575,0],
                        [-4.13157,-10.7417,-3.784,0]],
                    [ [-5.24568,-25.6688,-57.737,88.639], # N = 6
                        [-4.70693,-16.9178,-17.492,60.007],
                        [-4.42501,-13.1875,-5.104,27.877]],
                    [ [-5.51233,-29.5760,-69.398,164.295],# N = 7
                        [-4.97684,-19.9021,-22.045,110.761],
                        [-4.69648,-15.7315,-5.104,27.877]],
                    [ [-5.76202,-33.5258,-82.189,256.289], # N = 8
                        [-5.22924,-23.0023,-24.646,144.479],
                        [-4.95007,-18.3959,-7.344,94.872]],
                    [ [-5.99742,-37.6572,-87.365,248.316],# N = 9
                        [-5.46697,-26.2057,-26.627,176.382],
                        [-5.18897,-21.1377,-9.484,172.704]],
                    [ [-6.22103,-41.7154,-102.680,389.33],# N = 10
                        [-5.69244,-29.4521,-30.994,251.016],
                        [-5.41533,-24.0006,-7.514,163.049]],
                    [ [-6.43377,-46.0084,-106.809,352.752],# N = 11
                        [-5.90714,-32.8336,-30.275,249.994],
                        [-5.63086,-26.9693,-4.083,151.427]],
                    [ [-6.63790,-50.2095,-124.156,579.622],# N = 12
                        [-6.11279,-36.2681,-32.505,314.802],
                        [-5.83724,-29.9864,-2.686,184.116]]]
        tau_c_2010 = np.asarray(tau_c_2010)

        tau_ct_2010 = [[ [-3.95877,-9.0531,-28.428,-134.155],   # N = 1
                        [-3.41049,-4.3904,-9.036,-45.374],
                        [-3.12705,-2.5856,-3.925,-22.380]],
                    [ [-4.32762,-15.4387,-35.679,0],         # N = 2
                        [-3.78057,-9.5106,-12.074,0],
                        [-3.49631,-7.0815,-7.538,21.892]],
                    [ [-4.66305,-18.7688,-49.793,104.244],   # N = 3
                        [-4.11890,-11.8922,-19.031,77.332],
                        [-3.83511,-9.0723,-8.504,35.403]],
                    [ [-4.96940,-22.4694,-52.599,51.314],    # N = 4
                        [-4.42871,-14.5876,-18.228,39.647],
                        [-4.14633,-11.2500,-9.873,54.109]],
                    [ [-5.25276,-26.2183,-59.631,50.646],    # N = 5
                        [-4.71537,-17.3569,-22.660,91.359],
                        [-4.43422,-13.6078,-10.238,76.781]],
                    [ [-5.51727,-29.9760,-75.222,202.253],   # N = 6
                        [-4.98228,-20.3050,-25.224,132.03],
                        [-4.70233,-16.1253,-9.836,94.272]],
                    [ [-5.76537,-33.9165,-84.312,245.394],   # N = 7
                        [-5.23299,-23.3328,-28.955,182.342],
                        [-4.95405,-18.7352,-10.168,120.575]],
                    [ [-6.00003,-37.8892,-96.428,335.92],    # N = 8
                        [-5.46971,-26.4771,-31.034,220.165],
                        [-5.19183,-21.4328,-10.726,157.955]],
                    [ [-6.22288,-41.9496,-109.881,466.068],  # N = 9
                        [-5.69447,-29.7152,-33.784,273.002],
                        [-5.41738,-24.2882,-8.584,169.891]],
                    [ [-6.43551,-46.1151,-120.814,566.823],  # N = 10
                        [-5.90887,-33.0251,-37.208,346.189],
                        [-5.63255,-27.2042,-6.792,177.666]],
                    [ [-6.63894,-50.4287,-128.997,642.781],  # N = 11
                        [-6.11404,-36.4610,-36.246,348.554],
                        [-5.83850,-30.1995,-5.163,210.338]],
                    [ [-6.83488,-54.7119,-139.800,736.376],  # N = 12
                        [-6.31127,-39.9676,-37.021,406.051],
                        [-6.03650,-33.2381,-6.606,317.776]]]
        tau_ct_2010 = np.asarray(tau_ct_2010)

        # Don't have trend squared model rn TODO implement
        tau_ctt_2010 = [[ [-4.37113,-11.5882,-35.819,-334.047], # N = 1
                        [-3.83239,-5.9057,-12.490,-118.284],
                        [-3.55326,-3.6596,-5.293,-63.559]],
                        [ [-4.69276,-20.2284,-64.919,88.884],   # N =2
                        [-4.15387,-13.3114,-28.402,72.741],
                        [-3.87346,-10.4637,-17.408,66.313]],
                        [ [-4.99071,-23.5873,-76.924,184.782],  # N = 3
                        [-4.45311,-15.7732,-32.316,122.705],
                        [-4.17280,-12.4909,-17.912,83.285]],
                        [ [-5.26780,-27.2836,-78.971,137.871],  # N = 4
                        [-4.73244,-18.4833,-31.875,111.817],
                        [-4.45268,-14.7199,-17.969,101.92]],
                        [ [-5.52826,-30.9051,-92.490,248.096],  # N = 5
                        [-4.99491,-21.2360,-37.685,194.208],
                        [-4.71587,-17.0820,-18.631,136.672]],
                        [ [-5.77379,-34.7010,-105.937,393.991], # N = 6
                        [-5.24217,-24.2177,-39.153,232.528],
                        [-4.96397,-19.6064,-18.858,174.919]],
                        [ [-6.00609,-38.7383,-108.605,365.208], # N = 7
                        [-5.47664,-27.3005,-39.498,246.918],
                        [-5.19921,-22.2617,-17.910,208.494]],
                        [ [-6.22758,-42.7154,-119.622,421.395], # N = 8
                        [-5.69983,-30.4365,-44.300,345.48],
                        [-5.42320,-24.9686,-19.688,274.462]],
                        [ [-6.43933,-46.7581,-136.691,651.38],  # N = 9
                        [-5.91298,-33.7584,-42.686,346.629],
                        [-5.63704,-27.8965,-13.880,236.975]],
                        [ [-6.64235,-50.9783,-145.462,752.228], # N = 10
                        [-6.11753,-37.056,-48.719,473.905],
                        [-5.84215,-30.8119,-14.938,316.006]],
                        [ [-6.83743,-55.2861,-152.651,792.577], # N = 11
                        [-6.31396,-40.5507,-46.771,487.185],
                        [-6.03921,-33.8950,-9.122,285.164]],
                        [ [-7.02582,-59.6037,-166.368,989.879], # N = 12
                        [-6.50353,-44.0797,-47.242,543.889],
                        [-6.22941,-36.9673,-10.868,418.414]]]
        tau_ctt_2010 = np.asarray(tau_ctt_2010)

        coeffs = eval(f'tau_{model_type}_2010[{N-1}, {dict([(0.01, 0), (0.05, 1), (0.10, 2)])[sig]}, :]')

        return coeffs

# There's also an ADF-GLS test?
def adfTest(series, criterion="aic", model='ct', sig=0.05):
    
    """
    Tests for stationarity/unit root in time series
    Creates an autoregressive model of lag p (chosen through information criterion) of form:
    Δyt = α + βt + γyt-1 + δ1Δyt-1 + ... + δpΔyt-p + εt
    
    :param series: Arraylike of the series to test for stationarity

    param model: int signifying which regression to fit
        model = nc: no constant, no trend
                c: constant only
                ct: constant and trend
                ctt: constant and quadratic trend TO BE IMPLEMENTED
    """
    
    series = series.squeeze()

    if len(series.shape) >= 2:
        raise ValueError("Series should be 1D")
    
    if sig not in [0.01, 0.05, 0.10]:
        raise ValueError("Significance level should be 0.01, 0.05, or 0.10")

    if model not in ['nc', 'c', 'ct', 'ctt']:
            raise ValueError("Param model should be from values ['nc', 'c', 'ct', 'ctt']")
    
    # Fit OLS model to differenced series, using information criterion
    # Best_model is a tuple, (y, beta_hat, X, criterion score)
    best_model = None

    # Schwert/Ng-Perron rule?
    diff_series = np.diff(series)
    diff_len = len(diff_series)

    # Matrices for OLS
    # Matrices are in ascending order
    p_max = math.floor(12 * math.pow(len(series)/10, 0.25))  # Schwert rule
    for p in range(1, p_max+1):
        if model == 'nc': # no constant no drift
            
            X = np.empty(shape=(diff_len-p, p+1), dtype=float)
            X[:, 0] = series[p:-1]
            for i in range(1, p+1):
                X[:, i] = diff_series[p-i: diff_len-i]

            y = np.expand_dims(diff_series[p:], axis=-1)

        elif model == 'c': # constant only
            X = np.empty(shape=(diff_len-p, p+2), dtype=float)
            X[:, 0] = series[p:-1]
            X[:, 1] = np.ones(shape=(diff_len-p))
            for i in range(1, p+1):
                X[:, i+1] = diff_series[p-i: diff_len-i]
            
            y = np.expand_dims(diff_series[p:], axis=-1)

        elif model == 'ct':
            X = np.empty(shape=(diff_len-p, p+3))
            X[:, 0] = series[p:-1]
            X[:, 1] = np.ones(shape=(diff_len-p))
            X[:, 2] = np.arange(start=p+2, stop=diff_len+2)
            for i in range(1, p+1):
                X[:, i+2] = diff_series[p-i: diff_len-i]

            y = np.expand_dims(diff_series[p:], axis=-1)
        
        elif model == 'ctt':
            pass

        
        beta_hat = ols(y, X)

        # Calculate Loglik and AIC
        if best_model is None:
            best_model = (y, beta_hat, X, aic(loglike_ols(y, X, beta_hat), beta_hat.shape[0]))
        else:
            model_aic = aic(loglike_ols(y, X, beta_hat), beta_hat.shape[0])
            if model_aic < best_model[3]:
                best_model = (y, beta_hat, X, model_aic)

    gamma = best_model[1][0, 0]

    # Calculate critical value
    SSR = np.sum((best_model[0] - best_model[2]@best_model[1])**2)
    nobs = best_model[0].shape[0]
    resid_var = SSR/(nobs-best_model[1].shape[0])

    # LDL inv?
    cov = np.linalg.inv(best_model[2].T@best_model[2]) * resid_var
    gamma_SE = math.sqrt(cov[0, 0])

    print(f"DEBUG: Gamma coefficient: {gamma}")
    print(f"DEBUG: Gamma Standard Error: {gamma_SE}")
    t = gamma/gamma_SE


    mac_coeffs = mackinnon_crit_values(model, sig, N=1)
    crit_value = mac_coeffs[0] + (mac_coeffs[1]/nobs) + (mac_coeffs[2]/(nobs**2)) + (mac_coeffs[3]/(nobs**3))

    print(f"Test statistic: {t}, Critical value: {crit_value}")
    print(t<crit_value)
    if t < crit_value:
        print(f"At the {sig}% level, the series is stationary")
    else:
        print(f"At the {sig}% level, the series is not stationary")
    return best_model, t


def adfTable_monteCarlo():
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