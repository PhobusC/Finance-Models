import pandas as pd
import numpy as np
import tools



class SARIMAX(tools.MLEModel):
    """
    Class representing a SARIMAX model (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)
    """
    def __init__(self, endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.endog = endog
        self.exog = exog