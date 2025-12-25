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
            # 1e-8 prevents division by 0
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

        CHECK INVERTIBILITY
        
        """


        x0 = np.zeros(self.q + 1)
        x0[-1] = np.log(0.5 * np.var(self.data))
        bnds = [(-0.99, 0.99)] * self.q + [(-20, 20)]
        result = minimize(MA.kalman_negloglik, x0, args=(self.data, self.q), method='L-BFGS-B', bounds = bnds)
        
        if result.success:
            self.weights = np.insert(result.x[:self.q], 0, np.mean(self.data))
        else:
            print(f'Optimization failed: {result.message}')
            self.weights = result.x[:self.q]


        return self.weights
    
    def fit_mle(self):
        """
        Fits the MA model using conditional MLE
        """

    def predict(self, start: int, end: int):
        """
        Plots predictions using fitted model, starting from the specified indices
        """

        if(start > len(self.data) or end < 0 or start > end):
            raise ValueError()

        if(len(self.weights) == 0) :
            raise ValueError("Model must be fitted before prediction")
        
        if(self.q == 0):
            return np.full(end-start+1, self.weights[0])


        predictions = []
        resids = np.zeros(self.q)


        # Calculate residuals up to start then add to predictions
        for i in range(end+1):
            predict = self.weights[0] + np.dot(self.weights[1:], resids)

            # Add prediction if i is at least start
            if i >= start:
                predictions = np.append(predictions, predict)
            
            # Use data if available for innovations
            if i < len(self.data):
                resids = np.concatenate(([self.data[i]-predict], resids[:-1]))
            else:
                resids = np.concatenate(([0], resids[:-1]))
            
        return predictions
    
    def get_Weights(self):
        return self.weights
    
"""
data = np.array(pd.read_csv("daily_IBM.csv").close)

q = 0
model = MA(data, q)
weights = model.fit_Kalman()
start = 70
end = 90
predictions=model.predict(70, 90)


plt.plot(data, color='blue', label='Actual price')
plt.plot(np.linspace(start, end, num=end-start+1), predictions, 
        color='red', label = 'Predicted price')
plt.show()
"""


