import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.linalg import ldl
import pandas as pd
from tools import mackinnon_crit_values
from tools import adfTest
from statsmodels.tsa.stattools import adfuller



y = np.random.normal(loc=0, scale=1, size=(1000, 1))
adfstat, pvalue, usedlag, nobs, critvalues, _ = adfuller(y, regression='ct')
print(adfstat, pvalue, critvalues)
print()
model, t = adfTest(y)
