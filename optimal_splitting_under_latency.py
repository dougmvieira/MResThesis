from math import ceil
import numpy as np
import pandas as pd
from almgren_chriss import derived_params, almgren_chriss


np.seterr('raise')


def harmonic_sum(x, y):
    return 1/((1/x)+(1/y))


def model_with_latency(big_x, h, big_delta, big_t, sigma, lbda, eta, gamma):
    big_n = ceil((big_t-big_delta)/h)
    big_t_tilde = big_n*h - big_delta
    eta_tilde, kappa = derived_params(eta, gamma, h, lbda, sigma)

    impulse_exec_times = np.arange(big_n)*h + big_delta
    time_stamps = np.hstack((0, impulse_exec_times))

    almgren_chriss_big_y_1 = (big_x*np.sinh(kappa*big_t_tilde)
                              )/np.sinh(kappa*(big_t_tilde+h))
    if big_delta == h:
        big_y_1 = almgren_chriss_big_y_1
    else:
        adjust = (eta_tilde*big_x)/(h*lbda*(big_delta-h)*sigma**2)
        big_y_1 = harmonic_sum(almgren_chriss_big_y_1, adjust)

    inventory_values = np.hstack((big_x, big_y_1,
                                  almgren_chriss(big_y_1, big_n - 1,
                                                 big_t_tilde, sigma, lbda,
                                                 eta, gamma).values[1:]))

    return pd.Series(inventory_values, time_stamps)


def strategy_mean(y, h, big_delta, big_t, sigma, lbda, epsilon, eta, gamma):
    big_n = ceil((big_t-big_delta)/h)
    big_t_tilde = big_n*h - big_delta
    eta_tilde, kappa = derived_params(eta, gamma, h, lbda, sigma)

    almgren_chriss_big_y_1 = (y*np.sinh(kappa*big_t_tilde)
                              )/np.sinh(kappa*(big_t_tilde+h))
    if big_delta == h:
        big_y_1 = almgren_chriss_big_y_1
    else:
        adjust = (eta_tilde*y)/(h*lbda*(big_delta-h)*sigma**2)
        big_y_1 = harmonic_sum(almgren_chriss_big_y_1,
                               adjust)

    return (epsilon*y + gamma/2*y**2 + 2*eta_tilde/h*(y-big_y_1)**2
            + eta_tilde*big_y_1**2*(
                np.tanh(.5*kappa*h)*(h*np.sinh(2*kappa*big_t_tilde)
                                     + 2*big_t_tilde*np.sinh(kappa*h))
                / (np.sinh(kappa*big_t_tilde)**2*np.sinh(kappa*h))))


def strategy_var(y, h, big_delta, big_t, sigma, lbda, epsilon, eta, gamma):
    big_n = ceil((big_t - big_delta)/h)
    big_t_tilde = big_n*h - big_delta
    eta_tilde, kappa = derived_params(eta, gamma, h, lbda, sigma)

    inv_ac_big_y_1 = (np.sinh(kappa*(big_t_tilde+h))
                      / np.sinh(kappa*big_t_tilde))
    adjust = (h*lbda*(big_delta - h)*sigma**2)/eta_tilde
    big_y_1 = y/(inv_ac_big_y_1 + adjust)

    return big_delta*sigma**2*big_y_1 + (.5*sigma**2*big_y_1**2*(
             h*np.sinh(kappa*big_t_tilde)*np.cosh(kappa*(big_t_tilde - h))
             - big_t_tilde*np.sinh(kappa*h))
             / (np.sinh(kappa*big_t_tilde)**2*np.sinh(kappa*h)))
