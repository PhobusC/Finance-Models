import pandas as pd
import numpy as np

class AR():
    """
    Autoregressive Model
    Univariate
    Params: p - the number of lag observations included in the model
    """
    def __init__(self, p):
        self.weights = np.zeros((p, 1))
        self.p = p


    def autocovariance(self, data, lag: int) -> float:
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
    
    def autocov_fft(self, data) -> np.ndarray:
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
        fft_data = np.fft.fft(padded_data)
        power_spectrum = fft_data * np.conj(fft_data)
        autocov = np.fft.ifft(power_spectrum).real / n

        return autocov[:self.p+1]  # Return only up to lag p


    def fit_yule_walker(self, data):
        """
        Fits the AR model to the provided data using the Yule-Walker equations.
        
        Params: data - a list or array-like of historical data points. Shape (n, 1) (should only be one feature)
        http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/YWSourceFiles/YW-Eshel.pdf
        """

        # Calculate autocovariance and autocorrelation of order p
        # phi = R^-1 * r
        # phi is the weights
        # r calculated with autocovariance, divided by total variance
        if(len(data) <= self.p):
            raise ValueError("Data length must be greater than p")
        
        mean = np.mean(data)
        if(self.p == 0):
            return mean

        autocov = np.zeros((self.p, self.p))
        
        # Pad with 0s so len = 2^k (optional)
        """
        np_data = np.array(data)
        padded_length = 1 << len(np_data-1).bit_length()
        np_data = np.concatenate((np_data, np.zeros(padded_length - len(np_data))))
        """
        # Use FFT to compute autocovariance
        pass


    def fit_mle(self, data):
        """
        Fits the AR model to the provided data using Maximum Likelihood Estimation.
        
        Params: data - a list or array-like of historical data points. Shape (n, 1) (should only be one feature)
        """
        pass



data = pd.read_csv("daily_IBM.csv")
data_prices = data['close'].iloc[:50].values
model = AR(p=4)

print(model.autocov_fft(data_prices))
for lag in range(5):
    print(f"Lag {lag} autocovariance: {model.autocovariance(data_prices, lag)}")

print(data_prices.var())
    