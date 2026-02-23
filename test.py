import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.linalg import ldl
import pandas as pd
from tools import adfTest


y = np.random.random((50))
print(len(y))
print(y)
print("\n" * 3)
print(np.diff(y))
print("\n" * 3)

adfTest(y, model=3)

#print(np.arange(start=2, stop=11))