import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
"""
data = np.random.normal(0, 1, 50)

model = GenericLikelihoodModel(data)
print(model.loglike(np.random.normal(0, 1, 50)))
"""

y = np.eye(4)
print(y.shape)

print(np.array([4])@y)