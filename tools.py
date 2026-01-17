def adfTest(series):
    """
    Tests for stationarity/unit root in time series
    Creates an autoregressive model of lag p (chosen through information criterion) of form:
    Δyt = α + βt + γyt-1 + δ1Δyt-1 + ... + δpΔyt-p + εt
    
    :param series: Arraylike of the series to test for stationarity
    """
    


def aic():
    """
    Calculates the Akaike Information Criterion for the given model
    """


class KalmanFilter:
    """
    Represents a Kalman Filter with a given state-space representation.
    """
    def __init__(self, data):
        self.data = data

    def loglike(self):
        """
        Calculates loglike for the given state-space model
        """

        loglike = 0.0

        for i in 

        self.loglike = loglike
        return loglike

    def updateMatrices(self, matrices, initial_state, initial_cov):
        """
        Updates the transition matrices
        """
        self.T, self.R, self.Z, self.D = matrices
        self.initial_state = initial_state
        self.initial_cov = initial_cov
