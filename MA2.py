import numpy as np
from tools import MLEModel

class MA2(MLEModel):
    def __init__(self, endog, q=0):
        super().__init__(endog, k_states=q+1)
        self.q = q
        self._init_Repr()

    def _init_Repr(self):
        ssm = {}
        ssm["Z"] = np.zeros((1, self.filter.k_states))
        ssm["Z"][0, 0] = np.mean(self.endog)

        ssm["T"] = np.zeros((self.filter.k_states, self.filter.k_states))
        ssm["T"][0, 0] = 1
        for i in range(q):
            ssm["T"][i+2, i+1] = 1

        ssm["R"] = np.zeros((self.filter.k_states, 1))
        ssm["R"][1, 0] = 1

        initial_state = np.zeros((self.filter.k_states, 1))
        initial_state[0, 0] = 1
        ssm['init_state'] = initial_state

        initial_cov = 1e6 * np.eye(self.filter.k_states)
        ssm['init_cov'] = initial_cov
    


        
        self.filter.setRepr(ssm)

    def fit():

        result = super().fit()
        return result
    
    def _init_params(self):
        x0 = np.zeros(self.q + 1)
        x0[-1] = np.log(0.5 * np.var(self.data))
        return x0

    def change_spec(self, params):
        """
        Changes the states space model for a MA model
        params should be in the form [theta_1, ... , theta_q, var_eps]
        """
        weights = np.concatenate(np.array[1], params[:-1])
        var = np.exp(params[-1]) * np.eye(self.filter.k_states)

        self.filter.setRepr({'Z': weights, 'Q': var})


        
    

