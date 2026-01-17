import pandas as pd
import numpy as np



class SARIMAX():
    """
    Class representing a SARIMAX model (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)
    """
    def __init__(self, data, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), exog=None):
        self.data = data
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog = exog