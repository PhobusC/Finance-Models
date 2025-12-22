from AR import AR
from MA import MA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        ar_pred = self.ar.predict(start, end)
        ma_pred = self.ma.predict(start, end)
        print(ar_pred)
        print(ma_pred)
        # Un-difference series
        predictions = ar_pred + ma_pred - np.mean(self.ar.data)

        # Redifference to find first values
        # Alternatively, for i in [self.d-1, ... 0], find ith difference of original series 
        #       using binomal expanison

        diff = self.data.copy()
        first_vals = [diff[0]]

        for _ in range(1, self.d):
            diff = np.diff(diff)
            first_vals.append(diff[0])

        
        for i in range(self.d):
            predictions = np.cumsum(predictions) + first_vals[self.d-i-1]
        
        

        return predictions
    

data = pd.read_csv("daily_IBM.csv")
data_prices = data['close'].values


model = ARIMA(data_prices, 0, 0, 1)
model.fit()
print(np.mean(data_prices))
print(model.ar.weights)
print(model.ma.weights)
predictions = model.predict(70, 90)
print(predictions.shape)
plt.plot(np.linspace(70, 90, 21), predictions, color='red', label='Prediction')
plt.plot(data_prices, color='blue', label='Actual prices')
plt.legend()

plt.show()


