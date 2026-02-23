import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.linalg import ldl
import pandas as pd
from tools import adfTest


y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

max_lag=4
lag = 2
print(y[-(lag+1):-(11-max_lag+lag)])

for i in range(max_lag+1):
    print(y[max_lag-i: 10-i])