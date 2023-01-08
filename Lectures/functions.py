import numpy as np

def compute_pol(x,m):
    n = len(x) # number of points
    M = np.zeros((n,m))
    M[:,0] = np.ones((n,1)).T

    for i in range(1,m):
        M[:,i] = x ** i

    return M

def pol_interp(x,f,xnew):
    m = len(x) # number of nodes
    M0 = compute_pol(x,m)
    lamb = np.linalg.solve(M0, f)
    M1 = compute_pol(xnew,m)

    return M1 @ lamb

def pol_interp_ols(x,f,xnew,m):
    M0 = compute_pol(x,m)
    lamb = np.linalg.inv(M0.T @ M0) @ M0.T @ f
    M1 = compute_pol(xnew,m)

    return M1 @ lamb

def gy(y,x,f):
    a = x[0]
    b = x[1]
    c = x[2]
    fa=f[0]
    fb=f[1]
    fc=f[2]
    return (y-fa)*(y-fb)*c/((fc-fa)*(fc-fb)) + (y-fb)*(y-fc)*a/((fa-fb)*(fa-fc)) + (y-fc)*(y-fa)*b/((fb-fc)*(fb-fa))