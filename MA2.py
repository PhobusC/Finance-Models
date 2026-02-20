import numpy as np
import pandas as pd
from tools import MLEModel
from MA import MA

class MA2(MLEModel):
    def __init__(self, endog, q=0):
        super().__init__(endog, k_states=q+1, k_posdef=None) # TODO change k_posdef
        self.q = q
        self._init_Repr()

    def _init_Repr(self):
        ssm = {}
        """ Not necessary since we init this in _init_params
        ssm["Z"] = np.zeros((1, self.filter.k_states))
        ssm["Z"][0, 0] = np.mean(self.endog)
        """
        #TODO redo the initialization, this ssm is broken
        ssm["T"] = np.zeros((self.filter.k_states, self.filter.k_states))
        ssm["T"][0, 0] = 1
        for i in range(1, self.q):
            ssm["T"][i+1, i] = 1

        ssm["d"] = np.zeros((1, 1))
        ssm["c"] = np.zeros((self.filter.k_states, 1))

        ssm["R"] = np.zeros((self.filter.k_states, self.filter.k_states))
        ssm["R"][1, 0] = 1

        ssm['H'] = np.zeros((1, 1))

        initial_state = np.zeros((self.filter.k_states, 1))
        initial_state[0, 0] = 1
        ssm['init_state'] = initial_state

        initial_cov = 1e6 * np.eye(self.filter.k_states)
        ssm['init_cov'] = initial_cov

        self.filter.setRepr(ssm)
    
    def _init_params(self):
        x0 = np.zeros(self.q + 1)
        x0[-1] = np.log(0.5 * np.var(self.endog))
        return x0

    def change_spec(self, params):
        """
        Changes the states space model for a MA model
        params should be in the form [theta_1, ... , theta_q, log(var_eps)]
        """
        print("\n" * 3)
        print(params)
        weights = np.concatenate((np.array([1]), params[:-1]))
        weights = weights[None, :]
        var = np.exp(params[-1]) * np.eye(self.filter.k_states)

        self.filter.setRepr({'Z': weights, 'Q': var})


        
data = pd.read_csv('daily_IBM.csv')
data_prices = np.array(data[['close']]).T
print(data_prices.shape)
model = MA2(data_prices, 2)
test_model = MA(data_prices, 2)
result = model.fit()



