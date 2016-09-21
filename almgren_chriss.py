import numpy as np
import pandas as pd


np.seterr('raise')

figure2_params = {'X':       10**6,         # shares
                  'T':       5,             # days
                  'N':       5,
                  'sigma':   0.95,          # dollars per share per day sqrt
                  'epsilon': 0.0625,        # dollars per share
                  'gamma':   2.5*10**(-7),  # dollars per share squared
                  'eta':     2.5*10**(-6),  # dollars per day
                  'lbda':    2*10**(-6)}


def _eta_tilde(eta, gamma, tau):
    return eta*(1 - (gamma*tau)/(2*eta))


def _kappa_tilde_squared(lbda, sigma, eta_tilde):
    return lbda*(sigma**2)/eta_tilde


def _kappa(kappa_tilde_squared, tau):
    return np.arccosh(kappa_tilde_squared*(tau**2)/2 + 1)/tau


def derived_params(eta, gamma, tau, lbda, sigma):
    eta_tilde = _eta_tilde(eta, gamma, tau)
    kappa_tilde_squared = _kappa_tilde_squared(lbda, sigma, eta_tilde)
    kappa = _kappa(kappa_tilde_squared, tau)

    return eta_tilde, kappa


def almgren_chriss(big_x, big_n, big_t, sigma, lbda, eta, gamma):
    tau = big_t/big_n
    eta_tilde, kappa = derived_params(eta, gamma, tau, lbda, sigma)

    t_j = np.linspace(0, big_t, big_n + 1)
    inventory_values = big_x*np.sinh(kappa*(big_t - t_j))/np.sinh(kappa*big_t)

    return pd.Series(inventory_values, t_j)


def almgren_chriss_example():
    big_x = figure2_params['X']
    big_n = figure2_params['N']
    big_t = figure2_params['T']
    sigma = figure2_params['sigma']
    eta = figure2_params['eta']
    gamma = figure2_params['gamma']
    lbda = figure2_params['lbda']

    trajectory = almgren_chriss(big_x, big_n, big_t, sigma, lbda, eta, gamma)
    trajectory.plot()
