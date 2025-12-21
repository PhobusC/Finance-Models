import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt



class MA():
    """
    Moving Average (MA) model for time series forecasting.
    Can use ACF to determine the order of the MA model.
    https://people.duke.edu/~rnau/411arim3.htm

    Specifies a model of the form: X_t = μ + ε_t + θ_1ε_(t-1) + θ_2ε_(t-2) + ... + θ_qε_(t-q)
    Only meaningful predictions up to q steps ahead
    """
    def __init__(self, data, q: int):
        self.data = np.array(data)
        self.q = q
        self.weights = np.array([])
        
    @staticmethod
    def kalman_negloglik(params, data, q):
        """
        Negative log-likelihood function for the MA model using the Kalman filter.
        https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf

        Params: params - array of parameters [θ_1, θ_2, ..., θ_q, log(var)]
                data - array of time series data
                q - order of the MA model

        Returns: double log-likelihood value
        """
        
        mean = np.mean(data)
        centered_data = data - mean
        
        state = np.zeros((q+1, 1)) # initial state
        
        R = np.zeros((q+1, 1)) # state noise vector
        R[0, 0] = 1

        transition = np.zeros((q+1, q+1)) # state transition matrix
        for i in range(q):
            transition[i+1, i] = 1

        # Get params
        weights = np.zeros((1, q+1))
        weights[0, 0] = 1
        weights[0, 1:] = params[:q]
        Q = np.exp(params[q]) + 1e-6 # state noise variance, log(var) to ensure positive variance

        # Initialize state covariance matrix
        stateCov = 1e6 * np.eye(q+1)

        logLik = 0.0

        # Kalman filter loop
        for i in range(len(data)):
            # Prediction step
            predicted_state = transition@state
            pred_y = weights@predicted_state
            pred_stateCov = transition@stateCov@transition.T + Q*R@R.T
            pred_yCov = max((weights@pred_stateCov@weights.T).item(), 1e-8) # + observation noise, not needed

            # Filtering & forecast
            kalmanGain = pred_stateCov@weights.T/pred_yCov
            state = predicted_state + kalmanGain@(centered_data[i]-pred_y)
            #stateCov = (np.eye(q) - kalmanGain@weights)@pred_stateCov   # Joseph form
            #stateCov= pred_stateCov - kalmanGain@pred_yCov@kalmanGain.TAlternative form/Joseph stabilized form
            stateCov = (np.eye(q+1)-kalmanGain@weights)@pred_stateCov@(np.eye(q+1) - kalmanGain@weights).T # Symmetric Joseph form

            # Optional smoothing

            # Negative log-likelihood
            logLik += 0.5 * (np.log(2*np.pi*pred_yCov) + (centered_data[i]-pred_y)**2/pred_yCov)


        
        return logLik.item()

    def fit_Kalman(self):
        """
        Fits the MA model using the Kalman filter.
        First turns the data into a state space model 
            and then applies the Kalman filter to estimate the parameters.

        
        """


        x0 = np.zeros(self.q + 1)
        x0[-1] = np.log(0.5 * np.var(self.data))
        bnds = [(None, None)] * self.q + [(-20, 20)]
        result = minimize(MA.kalman_negloglik, x0, args=(self.data, self.q), method='L-BFGS-B', bounds = bnds)
        
        if result.success:
            self.weights = result.x[:self.q]
        else:
            print(f'Optimization failed: {result.message}')
            self.weights = result.x[:self.q]


        return self.weights
    
    def fit_mle(self):
        """
        Fits the MA model using conditional MLE
        """

    def predict(self, start: int, end: int):
        if(len(self.weights) == 0):
            raise ValueError("Model is not fitted yet. Please call fit() method first.")
        
        if(start > len(self.data) or end < 0 or start > end):
            raise ValueError()
        

        # Placeholder for prediction logic
        mean = np.mean(self.data)
        predictions = []
        resids = np.zeros(self.q)

        for i in range(start):
            predict = mean + np.dot(self.weights, resids)
            resid = (self.data[i] - predict).item()
            
            resids = np.concatenate(([resid], resids[:-1]))

        j = end-start+1
        for i in range(j):
            predict = mean + np.dot(self.weights, resids)
            predictions = np.append(predictions, predict)
            
            resids = np.concatenate(([0], resids[:-1]))

        plt.plot(self.data, color='blue', label='Actual price')
        plt.plot(np.linspace(start, end, num=j), predictions, 
                 color='red', label = 'Predicted price')
        plt.show()

        return predictions
    
    def get_Weights(self):
        return self.weights
    
data = np.array(pd.read_csv("daily_IBM.csv").close)

q = 4
model = MA(data, q)
weights = model.fit_Kalman()
model.predict(80, 83)

