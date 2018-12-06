import matplotlib.pyplot as plt
import numpy as np

import tellurium as te

import basis_functions

def run_sindy(data, param_names, d=1, time_steps=1):
    """
    Runs sindy...
    """
    pass

if __name__ == '__main__':
    # test 1: m=2, no noise
    model1 = te.loada("""
        J0: -> m1; l1 - k1*P1
        J1: m1 ->; k2*m1
        J2: -> P1; k3*m1
        J4: P1 ->; k4*P1

        l1 = 10;
        k1 = 0.1;
        k2 = 0.1;
        k3 = 0.1;
        k4 = 0.1;
    """)
    results = model1.simulate(0, 100, 100)
    plt.plot(results['time'], results['[m1]'])
    plt.plot(results['time'], results['[P1]'])
    plt.show()

    data = results[:, 1:]
    basis_vectors, all_combs = basis_functions.generate_basis_functions(data, d=1)
    derivs = basis_functions.calculate_derivatives(data, regularize=True, time_steps=1, alph=1e-1)
    plt.plot(derivs)
    plt.show()

