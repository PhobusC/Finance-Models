import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize

class AR():
    """
    Autoregressive Model
    Univariate, covariance stationary
    Use PCAF to determine p
    https://people.duke.edu/~rnau/411arim3.htm
    Box-Jenkins?

    Specifies a model of the form: X_t = c + ϕ_1*X_(t-1) + ϕ_2*X_(t-2) + ... + ϕ_p*X_(t-p) + ε_t
    Use ADF to check for stationarity

    Params: p - the number of lag observations included in the model
    """
    def __init__(self, data, p: int):

        if(len(data) <= p):
            raise ValueError("Data length must be greater than p")
        
        self.data = data
        self.weights = np.array([])
        self.p = p

    def get_weights(self):
        return self.weights

    @staticmethod
    def autocovariance( data: np.ndarray, lag: int) -> float:
        """
        Simple autocovariance calculation of O(n^2)
        
        :param data: a list or array-like of historical data points. Shape (n, 1) (should only be one feature)
        :param lag: the lag value to calculate autocovariance for
        """
        n = len(data)
        mean = np.mean(data)

        s1 = data[:n-lag]
        s2 = data[lag:]
        cov = np.sum((s1 - mean) * (s2 - mean)) / n
        return cov
    
    @staticmethod
    def autocov_fft(data: np.ndarray, lag: int) -> np.ndarray:
        """
        Autocovariance calculation using fft

        Params: data - a list or array-like of historical data points. Shape (n, 1) (should only be one feature)

        Returns: autocovariance value for all lags (up to lag p)
        """
        n = len(data)
        mean = np.mean(data)
        centered_data = data - mean

        # Pad with zeros to next power of 2 for efficiency
        bit_length = (n - 1).bit_length()
        padded_length = 1 << bit_length
        padded_data = np.concatenate((centered_data, np.zeros(padded_length - n)))

        # FFT
        # Convolution theorem and Wiener-Khinchin theorem (study more)
        fft_data = np.fft.fft(padded_data)
        power_spectrum = fft_data * np.conj(fft_data)
        autocov = np.fft.ifft(power_spectrum).real / n

        return autocov[:lag+1]  # Return only up to lag p

    def fit_yule_walker(self):
        print("fitting")
        """
        Fits the AR model to the provided data using the Yule-Walker equations.
        
        Params: data - a list or array-like of historical data points. Shape (n, 1) (univariate)
        http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/YWSourceFiles/YW-Eshel.pdf

        Returns: weights - the fitted weights of the AR model
        """

        # Calculate autocovariance and autocorrelation of order p
        # phi = R^-1 * r
        # phi is the weights
        # r calculated with autocovariance, divided by total variance
        
        
        mean = np.mean(self.data)
        if(self.p == 0):
            self.weights = np.array([mean])
            return mean

        
        # Pad with 0s so len = 2^k (optional)
        """
        np_data = np.array(data)
        padded_length = 1 << len(np_data-1).bit_length()
        np_data = np.concatenate((np_data, np.zeros(padded_length - len(np_data))))
        """
        # Use FFT to compute autocovariance
        autocov = self.autocov_fft(self.data, self.p)
        #autocorr = autocov / autocov[0]  # normalize by variance

        R = np.zeros((self.p, self.p))
        r = np.zeros(self.p)

        for i in range(self.p):
            r[i] = autocov[i+1]  # lag i+1 autocov
            for j in range(self.p):
                R[i, j] = autocov[abs(i - j)]  # lag |i-j| autocov

        self.weights = np.linalg.solve(R, r)
        intercept = mean * (1 - np.sum(self.weights))
        self.weights = np.insert(self.weights, 0, intercept)  # insert intercept term
        return self.weights

    def fit_mle(self):
        """
        Fits the AR model to the provided data using Maximum Likelihood Estimation.

        scipy.optimize.minimize negative log-likelihood
        statsmodel inherit from GenericLikelihoodModel
        Or use OLS
        
        Params: data - a list or array-like of historical data points. Shape (n, 1) (should only be one feature)

        Returns: weights - the fitted weights of the AR model
        """

        Y = np.array(self.data[self.p:])
        """
        This does the same thing
        X = np.array(
            [np.concatenate(([1], data[::-1][len(data)-i-self.p:len(data)-i])) for i in range(len(data)-self.p)]
        )
        """

        X = np.zeros((len(self.data)-self.p, self.p+1))
        X[:, 0] = 1  # intercept column

        for i in range(len(self.data)-self.p):
            
            # X_(t-1), X_(t-2), ..., X_(t-p) for observation at time t
            X[i, 1:] = self.data[i:i+self.p][::-1]
            
        # OLS solution
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ Y

        return self.weights
    
    def predict(self, start: int, end: int) -> np.ndarray:
        """
        Predicts future values using the fitted AR model.
        
        Params: start - the starting index for prediction (inclusive)
                end - the ending index for prediction (inclusive)
        
        Returns: predictions - an array of predicted values from start to end
        """
        
        predictions = []
        data_extended = list(self.data)

        for t in range(start, end + 1):
            pred = np.dot(self.weights, np.concatenate(([1], data_extended[t - self.p:t][::-1])))
            predictions.append(pred)
            if t >= len(self.data):
                data_extended.append(pred)  # Append prediction for future predictions

        return np.array(predictions)


    


"""
# Testing

data = pd.read_csv("daily_IBM.csv")
data_prices = data['close'].values
mean_price = np.mean(data_prices)
model = AR(data_prices, p=4)
model.fit_mle()
# Center the data for predictions

predictions = model.predict(start=70, end=90)


model_sm = AutoReg(data_prices, lags=4).fit()
preds_sm = model_sm.predict(start=70, end=90)

print(model_sm.params)



plt.plot(np.linspace(70, 90, 21), preds_sm, label='Statsmodels AR Predictions', color='green')
plt.plot(np.linspace(70, 90, 21), predictions, label='AR Predictions', color='red')
plt.plot(data['close'].iloc[:100].values, label='Data', color='blue')
plt.title('AR Model Predictions vs Actual Data')
plt.show()
"""

    