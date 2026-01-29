import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
"""
data = np.random.normal(0, 1, 50)

model = GenericLikelihoodModel(data)
print(model.loglike(np.random.normal(0, 1, 50)))
"""

y = np.random.normal(0, 1, 50)
print(y.shape)

y@None