# based on https://github.com/sunju/psv/blob/master/ADM.m
import numpy as np

def soft_thresholding(X, d):
    X2 = np.abs(X) - d
    X2[X2 < 0] = 0
    Y = np.sign(X)*X2
    return Y

def norm(x):
    return np.sqrt(np.sum(x**2))

def adm(Y, q_init, lam, max_iter, tol):
    """
    runs the
    """
    q = q_init
    for k in range(max_iter):
        q_old = q
        x = soft_thresholding(Y*q, lam)
        q = np.dot(Y.T, x)/norm(np.dot(Y.T, x))
        res_q = norm(q_old - q)
        if res_q <= tol:
            return q
    return q
