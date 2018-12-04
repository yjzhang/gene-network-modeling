import itertools

import numpy as np

# basis functions for sindy - polynomial basis functions

def generate_basis_functions(X, d=3):
    """
    Args:
        X (array): data matrix of shape (time_steps, num_vars)
        d (int): maximum degree of polynomial

    Returns:
        ([1, X, X^2,...,X^d], [combinations])
    """
    output_vectors = []
    all_combs = []
    d1 = np.ones(X.shape[0])
    output_vectors.append(d1)
    all_combs.append((-1,))
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

def calculate_derivatives(X, regularize=False, time_steps=None):
    """
    Args:
        X (array): data matrix of shape (time_steps, num_vars)

    Returns:
        dX: array of dX/dt at all time points, in the same shape as X.
    """
    deltas = X[1:,:] - X[:-1,:]
    total_variation = np.sum(np.abs(deltas), 0)
    # TODO: do some sort of regularization?
    # see: https://github.com/stur86/tvregdiff
    return deltas

def sindy_lasso(X, dX, d=3):
    """
    Runs a lasso sparse
    """
    from sklearn.linear_model import Lasso
    basis_vectors, all_combs = generate_basis_functions(X, d)
    model = Lasso(fit_intercept=False)
    model.fit(basis_vectors, dX)
    return model
