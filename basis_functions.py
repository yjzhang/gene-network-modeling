import itertools

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV

# basis functions for sindy - polynomial basis functions

def generate_basis_functions(X, d=3):
    """
    Args:
        X (array): data matrix of shape (time_steps, num_vars)
        d (int): maximum degree of polynomial

    Returns:
        ([X, X^2,...,X^d], [combinations])
    """
    output_vectors = []
    all_combs = []
    #d1 = np.ones(X.shape[0])
    #output_vectors.append(d1)
    #all_combs.append((-1,))
    for dim in range(1, d+1):
        # Xd should be d-th degree polynomial basis functions on X
        combinations = itertools.combinations_with_replacement(range(X.shape[1]), dim)
        Xd = []
        for c in combinations:
            all_combs.append(c)
            Xc = X[:,c]
            Xc = Xc.prod(1)
            Xd.append(Xc)
        Xd = np.column_stack(Xd)
        output_vectors.append(Xd)
    output_vectors = np.column_stack(output_vectors)
    return output_vectors, all_combs

def generate_implicit_basis_functions(X, dX, d=3, constant=False):
    """
    Args:
        X (array): data matrix of shape (time_steps, num_vars)
        dX (vector): derivative for the given variable
        d (int): maximum degree of polynomial

    Returns:
        ([X, X^2,...,X^d, X'X, X'X^2,...], [combinations])
    """
    output_vectors = []
    output_derivs = []
    all_combs = []
    if constant:
        d1 = np.ones(X.shape[0])
        output_vectors.append(d1)
        all_combs.append(('c',))
    for dim in range(1, d+1):
        # Xd should be d-th degree polynomial basis functions on X
        combinations = itertools.combinations_with_replacement(range(X.shape[1]), dim)
        Xd = []
        for c in combinations:
            all_combs.append(c)
            Xc = X[:,c]
            Xc = Xc.prod(1)
            Xd.append(Xc)
            X_deriv = dX*Xc
            output_derivs.append(X_deriv)
        Xd = np.column_stack(Xd)
        output_vectors.append(Xd)
    output_vectors = np.column_stack(output_vectors + output_derivs)
    all_combs = all_combs + all_combs
    return output_vectors, all_combs


def calculate_derivatives(X, regularize=False, time_steps=1.0, alph=10):
    """
    Args:
        X (array): data matrix of shape (time_steps, num_vars)

    Returns:
        dX: array of dX/dt at all time points, in the same shape as X.
    """
    deltas = X[1:,:] - X[:-1,:]
    # TODO: do some sort of regularization?
    # see: https://github.com/stur86/tvregdiff
    if regularize:
        import tvregdiff
        deltas = []
        for x in range(X.shape[1]):
            Xc = X[:,x]
            deltas.append(tvregdiff.TVRegDiff(Xc, 100, alph=alph, dx=time_steps))
        deltas = np.column_stack(deltas)
    return deltas

def sindy_lasso(X, dX, d=3, alpha=0.1):
    """
    Runs a lasso sparse regression...
    """
    basis_vectors, all_combs = generate_basis_functions(X, d)
    model = Lasso(fit_intercept=True, alpha=alpha)
    model.fit(basis_vectors, dX)
    return model, all_combs

def run_protein():

    data_mrna = pd.read_csv('Scipio_Wild_0_MassSpec.csv')
    data_array = data_mrna.iloc[:, 1:].values
    time_step = data_mrna.time[1] - data_mrna.time[0]
    deriv = calculate_derivatives(data_array, True, time_step)

    #import matplotlib.pyplot as plt
    #plt.plot(deriv)
    #plt.show()

    basis_vectors, all_combs = generate_basis_functions(data_array, 3)
    model = Lasso(fit_intercept=True, alpha=0.1)
    model.fit(basis_vectors, deriv[1:,:])
    predicted_derivs = model.predict(basis_vectors)
    print(predicted_derivs)
    error = np.sum((predicted_derivs - deriv[1:,:])**2)
    print(error)
    return model, basis_vectors, all_combs, data_mrna, deriv

def run_mrna():

    data_mrna = pd.read_csv('Scipio_Wild_0_RNASeq.csv')
    data_array = data_mrna.iloc[:, 1:].values
    time_step = data_mrna.time[1] - data_mrna.time[0]
    deriv = calculate_derivatives(data_array, True, time_step)

    #import matplotlib.pyplot as plt
    #plt.plot(deriv)
    #plt.show()

    basis_vectors, all_combs = generate_basis_functions(data_array, 3)
    model = LassoCV(fit_intercept=True)
    model.fit(basis_vectors, deriv[1:,:])
    predicted_derivs = model.predict(basis_vectors)
    print(predicted_derivs)
    error = np.sum((predicted_derivs - deriv[1:,:])**2)
    print(error)
    return model, basis_vectors, all_combs, data_mrna, deriv


def run_protein_mrna():


    data_mrna = pd.read_csv('Scipio_Wild_0_RNASeq.csv')
    data_protein = pd.read_csv('Scipio_Wild_0_MassSpec.csv')
    data_mrna['input'] = [1]*data_mrna.shape[0]
    all_vars = data_mrna.columns[1:].append(data_protein.columns[1:])
    print(all_vars)
    data_array = data_mrna.iloc[::2, 1:].values
    data_p_array = data_protein.iloc[:, 1:].values
    data_array = np.hstack([data_array, data_p_array])
    time_step = data_protein.time[1] - data_protein.time[0]
    deriv = calculate_derivatives(data_array, True, time_step, alph=100)

    import matplotlib.pyplot as plt
    plt.plot(deriv)
    plt.show()

    basis_vectors, all_combs = generate_basis_functions(data_array, d=2)
    all_var_combs = [all_vars[list(c)].tolist() for c in all_combs]
    #model = MultiTaskLassoCV(fit_intercept=True)
    model = Lasso(fit_intercept=False, alpha=0.01)
    model.fit(basis_vectors, deriv[1:,:])
    predicted_derivs = model.predict(basis_vectors)
    print(predicted_derivs)
    error = np.sum((predicted_derivs - deriv[1:,:])**2)
    print(error)
    return model, basis_vectors, all_var_combs, all_vars, all_combs, data_mrna, data_protein, deriv


if __name__ == '__main__':
    """
    from sklearn.linear_model import Lasso
    import pandas as pd

    data_mrna = pd.read_csv('Scipio_Wild_0_RNASeq.csv')
    data_array = data_mrna.iloc[:, 1:].values
    time_step = data_mrna.time[1] - data_mrna.time[0]
    deriv = calculate_derivatives(data_array, True, time_step)

    import matplotlib.pyplot as plt
    plt.plot(deriv)
    plt.show()

    basis_vectors, all_combs = generate_basis_functions(data_array, 3)
    model = Lasso(fit_intercept=False, alpha=0.01)
    model.fit(basis_vectors, deriv[1:,:])
    predicted_derivs = model.predict(basis_vectors)
    """

    both_results = run_protein_mrna()
    model = both_results[0]
    params = model.coef_
    intercept = model.intercept_
    all_var_combs = both_results[2]
    all_vars = both_results[3]
    nonzero_params = np.nonzero(params)
    param_equations = {v:[] for v in all_vars}
    for x, y in zip(*nonzero_params):
        print(x, y)
        print(all_var_combs[y])
        print(all_vars[x])
        print(params[x, y])
        param_equations[all_vars[x]].append((all_var_combs[y], params[x,y]))
    # convert param_equations to tellurium/antimony format...
    # Ji: -> Pi; [eqs]
    index = 0
    antimony_eqs = ''
    antimony_vars = ''
    param_index = 0
    for var, params in param_equations.items():
        positive_params = [p for p in params if p[1] > 0]
        positive_rate = ' + '.join(str(x[1]) + '*' + '*'.join(x[0]) for x in positive_params)

        negative_params = [p for p in params if p[1] < 0]
        negative_rate = ' + '.join(str(-x[1]) + '*' + '*'.join(x[0]) for x in negative_params)
#        if intercept[index] > 0:
#            positive_rate = str(intercept[index]) + ' + ' + positive_rate
#        else:
#            negative_rate = str(-intercept[index]) + ' + ' + negative_rate

        antimony_eqs += '\nJ{0}: -> {1}; {2}'.format(index*2, var, positive_rate)
        antimony_eqs += '\nJ{0}: {1} ->; {2}'.format(index*2+1, var, negative_rate)
        index += 1
    print(antimony_eqs)
