import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

import tellurium as te

import basis_functions

def run_sindy(data, param_names, d=1, time_steps=1, alph=1e-1, plot=False,
        alpha=1e-2, fit_intercept=True):
    """
    Runs sindy to generate a system of ODEs that represent the given data...
    """
    basis_vectors, all_combs = basis_functions.generate_basis_functions(data, d=d)
    param_names = np.array(param_names)
    all_var_combs = [param_names[list(c)].tolist() for c in all_combs]
    derivs = basis_functions.calculate_derivatives(data, regularize=True,
            time_steps=time_steps, alph=alph)
    if plot:
        plt.title('derivs')
        plt.plot(derivs)
        plt.show()
    model = Lasso(fit_intercept=fit_intercept, alpha=alpha)
    model.fit(basis_vectors, derivs[1:,:])
    params = model.coef_
    intercept = model.intercept_
    # calculate that one thing...
    nonzero_params = np.nonzero(params)
    param_equations = {v:[] for v in param_names}
    for x, y in zip(*nonzero_params):
        print(x, y)
        print(all_var_combs[y])
        print(param_names[x])
        print(params[x, y])
        param_equations[param_names[x]].append((all_var_combs[y], params[x,y]))
    index = 0
    antimony_eqs = ''
    antimony_vars = ''
    param_index = 0
    for var, params in param_equations.items():
        positive_params = [p for p in params if p[1] > 0]
        positive_rate = ' + '.join(str(x[1]) + '*' + '*'.join(x[0]) for x in positive_params)

        negative_params = [p for p in params if p[1] < 0]
        negative_rate = ' + '.join(str(-x[1]) + '*' + '*'.join(x[0]) for x in negative_params)
        try:
            if intercept[index] > 0:
                if len(positive_rate) == 0:
                    positive_rate = str(intercept[index])
                else:
                    positive_rate = ' + '.join([str(intercept[index]), positive_rate])
            else:
                if len(negative_rate) == 0:
                    negative_rate = str(intercept[index])
                else:
                    negative_rate = ' + '.join([str(-intercept[index]), negative_rate])
        except:
            pass

        antimony_eqs += '\nJ{0}: -> {1}; {2}'.format(index*2, var, positive_rate)
        antimony_eqs += '\nJ{0}: {1} ->; {2}'.format(index*2+1, var, negative_rate)
        index += 1
    print(antimony_eqs)
    return antimony_eqs



if __name__ == '__main__':
    # test 1: 3-node IFF
    model1 = te.loada("""
        J0: -> g1; k1 - k2*g3 - k7*g1
        J1: -> g2; k3 + k4*g1 - k8*g2
        J2: -> g3; k5 + k6*g2 - k9*g3

        k1 = 1.1;
        k2 = 0.1;
        k3 = 1.1;
        k4 = 0.1;
        k5 = 1.1;
        k6 = 0.1;
        k7 = 0.1;
        k8 = 0.1;
        k9 = 0.1;
    """)
    results = model1.simulate(0, 100, 100)
    data = results[:, 1:]
    plt.plot(results['time'], results['[g1]'])
    plt.plot(results['time'], results['[g2]'])
    plt.plot(results['time'], results['[g3]'])
    plt.show()

    param_names = ['g1', 'g2', 'g3']

    antimony_eqs = run_sindy(data, param_names, alpha=1e-2, alph=1, plot=True, fit_intercept=True)
    model2 = te.loada(antimony_eqs)
    results_2 = model2.simulate(0, 100, 100)
    data_2 = results_2[:, 1:]

    error = np.sqrt(np.mean((data_2 - data)**2))
    print(error)

    #basis_vectors, all_combs = basis_functions.generate_basis_functions(data, d=1)
    #derivs = basis_functions.calculate_derivatives(data, regularize=True, time_steps=1, alph=1e-1)
    #plt.plot(derivs)
    #plt.show()

