import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.linalg import ldl
import pandas as pd
from tools import mackinnon_crit_values
from tools import adfTest
from tools import aic
from tools import loglike_ols
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS



y = np.random.normal(loc=0, scale=1, size=(1000, 1))
adfstat, pvalue, usedlag, nobs, critvalues, _ = adfuller(y, regression='ct')
print(adfstat, usedlag)
print()
model, t = adfTest(y, model='ct')
#print(model[2][:, 0:5])
print(model[2].shape)

result = OLS(model[0], model[2]).fit()
print(f"True llk: {result.llf}")
print(f"Calculated llk: {loglike_ols(model[0], model[2], model[1])}")

print(result.params)
print(model[1])