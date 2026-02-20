import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.linalg import ldl
"""
data = np.random.normal(0, 1, 50)

model = GenericLikelihoodModel(data)
print(model.loglike(np.random.normal(0, 1, 50)))
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
