from sindy_experiments import run_implicit_sindy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

import tellurium as te

if __name__ == '__main__':
    # test 1: 3-node model from class on 2018-10-26
    model1 = te.loada("""
            J1: -> A ; v0
            J2: A -> B ; ka*A
            J3: B -> C ; kb*B
            J4: C -> ; kc*C

            v0 = 10;
            ka = 0.4;
            kb = 0.32;
            kc = 0.4;
    """)
    results = model1.simulate(0, 50, 50)
    data = np.column_stack([results['[A]'], results['[B]'], results['[C]']])
    data_no_noise = data.copy()

    param_names = ['A', 'B', 'C']
    noise_variance = 0

    # add noise to simulated model
    rmse_vals = {}

    noise = np.random.randn(*data.shape)*noise_variance
    data += noise
    params = run_implicit_sindy(data, param_names, alph=1e-2, lam=1e-1)

