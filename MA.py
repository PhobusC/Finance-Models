import numpy as np
import pandas as pd
from scipy.optimize import minimize



class MA():
    """
    Moving Average (MA) model for time series forecasting.
    Can use ACF to determine the order of the MA model.
    https://people.duke.edu/~rnau/411arim3.htm

    Specifies a model of the form: X_t = μ + ε_t + θ_1ε_(t-1) + θ_2ε_(t-2) + ... + θ_qε_(t-q)
    Only meaningful predictions up to q steps ahead
    """
    def __init__(self, data, q: int):
        self.data = data
        self.q = q
        self.weights = np.array([])
        
    @staticmethod
    def kalman_loglik(params, data, q):
        """
        Log-likelihood function for the MA model using the Kalman filter.
        https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf

        Params: params - array of parameters [θ_1, θ_2, ..., θ_q]
                data - array of time series data
                q - order of the MA model

        Returns: double log-likelihood value
        """

        
        
        state = np.zeros((q, 1)) # initial state

        mean = np.mean(data)

        centered_data = data - mean
        R = np.zeros((q, 1)) # state noise vector
        R[0, 0] = 1

        transition = np.zeros((q, q)) # state transition matrix
        for i in range(q-1):
            transition[i+1, i] = 1

        # Get params
        weights = params[:q]
        Q = params[q] # state noise variance

        # Initialize state covariance matrix
        stateCov = 1e6 * np.eye(q)
        


        logLik = 0.0

        # Kalman filter loop
        for i in range(len(data)):
            # Prediction step
            predicted_state = transition@state
            pred_y = weights@predicted_state
            pred_stateCov = transition@stateCov@transition.T + Q*R@R.T
            pred_yCov = weights@pred_stateCov@weights.T # + observation noise, not needed

            # Filtering & update
            kalmanGain = pred_stateCov@weights.T@np.linalg.inv(pred_yCov)
            corrected_state = predicted_state + kalmanGain@(centered_data[i]-pred_y)
            #corrected_stateCov = (np.eye(q) - kalmanGain@weights)@pred_stateCov   # Joseph form
            corrected_stateCov = pred_stateCov - kalmanGain@pred_yCov@kalmanGain.T # Alternative form/Joseph stabilized form

            # Forecast
            state = corrected_state
            stateCov = corrected_stateCov

            # Optional smoothing


            # Log-likelihood
            logLik += 




        return 0

    def fit_Kalman(self):
        """
        Fits the MA model using the Kalman filter.
        First turns the data into a state space model 
            and then applies the Kalman filter to estimate the parameters.

        
        """


        x0 = np.zeros(self.q + 1)
        x0[-1] = 0.5 * np.var(self.data)

        bnds = ((None, None), (0, None))
        



            
            

        return self.weights
    
    

    def fit_mle(self):
        """
        Fits the MA model using conditional MLE
        """

    def predict(self, start: int, end: int):
        if(len(self.weights) == 0):
            raise ValueError("Model is not fitted yet. Please call fit() method first.")
        # Placeholder for prediction logic
        return [0] * (end - start + 1)
    
    def get_Weights(self):
        return self.weights