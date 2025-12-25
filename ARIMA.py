from AR import AR
from MA import MA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as statsARIMA

class ARIMA():
    def __init__(self, data, p: int, d: int, q: int):
        if p < 0 or d < 0 or q < 0:
            raise ValueError()
        
        
        self.data = data
        self.d = d

        d_series = np.diff(self.data, d)
        self.ar = AR(d_series, p)
        self.ma = MA(d_series, q)

    def fit(self):
        self.ar.fit_mle()
        self.ma.fit_Kalman()

    def predict(self, start: int, end: int):
        ar_pred = 0
        ma_pred = 0

        # Subtract self.d bc differencing
        ar_pred = self.ar.predict(start-self.d, end-self.d)
        ma_pred = self.ma.predict(start-self.d, end-self.d)

        # Un-difference series
        predictions = ar_pred + ma_pred - np.mean(self.ar.data)
        
        # Redifference to find first values
        # Alternatively, for i in [self.d-1, ... 0], find ith difference of original series 
        #       using binomal expanison

        diffs = [self.data]
        for _ in range(self.d):
            diffs.append(np.diff(diffs[-1]))

        
        first_vals = [diffs[k-1][start-k] for k in range(1, self.d+1)]

        
        for i in range(self.d):
            predictions = np.cumsum(predictions) + first_vals[self.d-i-1]
            
        
        

        return predictions
    

data = pd.read_csv("daily_IBM.csv")
data_prices = data['close'].values


model = ARIMA(data_prices, 0, 0, 1)
model.fit()

test_model = statsARIMA(data_prices, order=(0, 0, 1)).fit()

start = 70
end = 90
predictions = model.predict(start, end)
test_pred = test_model.predict(start=start, end=end)
plt.plot(np.linspace(start, end, end-start+1), predictions, color='red', label='Prediction')
plt.plot(np.linspace(start, end, end-start+1), test_pred,  color='purple', label='Statsmodel Prediction')
plt.plot(data_prices, color='blue', label='Actual prices')

plt.title("ARIMA predictions")
plt.legend()

plt.show()


