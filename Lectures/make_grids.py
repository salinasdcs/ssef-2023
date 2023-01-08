import numpy as np

def discretize_capital_exponential(kmin, kmax, n_k):
    # Find maximum ubar 
    ubar = np.log(1 + np.log(1 + kmax - kmin))
    
    # Make uniform grid
    u_grid = np.linspace(0, ubar, n_k)
    
    # Double-exponentiate uniform grid 
    return kmin + np.exp(np.exp(u_grid) - 1) - 1

def stationary_markov(Pi, tol=1E-14):
    # Start with uniform distribution over all states
    N = Pi.shape[0]
    pi = np.full(N, 1/N)
    
    # Update distribution using Pi until successive iterations differ by less than tol
    for _ in range(10_000):
        pi_new = Pi.T @ pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new

def rouwenhorst_Pi(N, p):
    # Step 1
    Pi = np.array([[p, 1 - p],
                   [1 - p, p]])
    
    # Step 2
    for n in range(3, N + 1):
        Pi_old = Pi
        Pi = np.zeros((n, n))
        
        Pi[:-1, :-1] += p * Pi_old
        Pi[:-1, 1:] += (1 - p) * Pi_old
        Pi[1:, :-1] += (1 - p) * Pi_old
        Pi[1:, 1:] += p * Pi_old
        Pi[1:-1, :] /= 2
        
    return Pi

def markov_rouwenhorst(rho_z, sigma_z, n_z):
    # Choose inner-switching probability p to match persistence rho_z
    p = (1+rho_z)/2
    
    # Start with states from 0 to n_z-1, scale by alpha to match standard deviation sigma_z
    s = np.arange(n_z)
    alpha = 2*sigma_z/np.sqrt(n_z-1)
    s = alpha*s
    
    # Obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_z, p)
    pi = stationary_markov(Pi)
    
    # z is log tfp, get tfp z and scale so that mean is 1
    y = np.exp(s)
    y /= np.vdot(pi, y)
    
    return y, pi, Pi