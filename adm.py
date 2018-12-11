# based on https://github.com/sunju/psv/blob/master/ADM.m
import numpy as np
import scipy.linalg

def soft_thresholding(X, d):
    X2 = np.abs(X) - d
    X2[X2 < 0] = 0
    Y = np.sign(X)*X2
    return Y

def norm(x):
    return np.sqrt(np.sum(x**2))

def gs(X):
    Q, R = scipy.linalg.qr(X)
    return Q

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def adm(Y, q_init=None, lam=1e-2, max_iter=100, tol=1e-10):
    """
    runs the ADM algorithm for finding the sparsest vector in the null space
    of Y.
    """
    if q_init is None:
        best_objective = 1e10
        best_q = None
        for i in range(Y.shape[0]):
            q_init = Y[i,:]
            q_init = q_init/norm(q_init)
            q = q_init
            for k in range(max_iter):
                q_old = q
                x = soft_thresholding(np.dot(Y, q), lam)
                q = np.dot(Y.T, x)
                q = q/norm(q)
                res_q = norm(q_old - q)
                objective = (q != 0).sum()
                print(objective)
                if res_q <= tol:
                    break
            if objective < best_objective:
                best_objective = objective
                best_q = q
        # solve a linear program...
        return best_q

if __name__ == '__main__':
    X = np.random.rand(10,50)
    for x, y in zip(np.random.randint(0, 10, 10), np.random.randint(0, 50, 10)):
        X[x,y] = 0
    Q = scipy.linalg.null_space(X)
    result = adm(Q, lam=1e-2)
    print(result)
    print(np.abs(np.dot(X, np.dot(Q, result))).sum())
