import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.linalg import ldl
import pandas as pd
"""
data = np.random.normal(0, 1, 50)

model = GenericLikelihoodModel(data)
print(model.loglike(np.random.normal(0, 1, 50)))
"""
"""
y = np.random.random((3, 3))
y = y + y.T
print(y)

l, d, perm = ldl(y, lower=False)
print(l)
print(d)
print(perm)
print(l[perm])

print(l@d@l.T)
"""
"""
data = pd.read_csv("Nile.csv")
data = data.set_index('time')
data = data['Nile']

print(data.iloc[:10])

print(data.isna().sum())


print(np.concat((np.empty(0), np.array([1, 2, 3]))))
"""


y = np.array([True, True, False, False, False, True])
print(np.sum(y))

