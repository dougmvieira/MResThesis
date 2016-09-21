from math import ceil
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from almgren_chriss import figure2_params, derived_params, almgren_chriss
from utils import days_to_seconds, miliseconds_to_days


np.seterr('raise')


used_params = {'y':       figure2_params['X'],
               'T':       miliseconds_to_days(1000),
               'sigma':   figure2_params['sigma'],
               'epsilon': figure2_params['epsilon'],
               'gamma':   figure2_params['gamma'],
               'eta':     figure2_params['eta']*miliseconds_to_days(10),
               'h':       miliseconds_to_days(10),
               'Delta':   miliseconds_to_days(200),
               'lbda':    0.1}


def print_used_params():
    names_and_params = [('Initial inventory',
                         '{}'.format(used_params['y'])),
                        ('Time horizon',
                         '{}s'.format(int(days_to_seconds(used_params['T'])))),
                        ('Volatility (daily)',
                         '{}'.format(used_params['sigma'])),
                        (r'Temporary impact ($\epsilon$)',
                         '{}'.format(used_params['epsilon'])),
                        (r'Temporary impact ($\eta/h$)',
                         '{:.5}'.format(used_params['eta']
                                        / miliseconds_to_days(10))),
                        ('Permanent impact',
                         '{}'.format(used_params['gamma'])),
                        ('Decision lag', '{}ms'.format(
                            int(1000*days_to_seconds(used_params['h'])))),
                        ('Execution delay', '{}ms'.format(
                            int(1000*days_to_seconds(used_params['Delta'])))),
                        ('Risk aversion',
                         '{}'.format(used_params['lbda']))]
    pd.DataFrame(names_and_params, columns=['Parameter', 'Value']
                 ).to_latex('used_params.tex', escape=False)


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


def inventory_example():
    y = used_params['y']
    h = used_params['h']
    big_delta = used_params['Delta']
    big_t = used_params['T']
    sigma = used_params['sigma']
    eta = used_params['eta']
    gamma = used_params['gamma']
    lbda = used_params['lbda']

    trajectory = model_with_latency(y, h, big_delta, big_t, sigma, lbda, eta,
                                    gamma)
    trajectory_without_latency = model_with_latency(y, h, h, big_t, sigma,
                                                    lbda, eta, gamma)

    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)
    ax.step(days_to_seconds(trajectory.index.values), trajectory.values,
            where='post', label=r'$\Delta$=200ms')
    ax.step(days_to_seconds(trajectory_without_latency.index.values),
            trajectory_without_latency.values, where='post',
            label=r'$\Delta$=0ms')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Size of the inventory')
    ax.set_ylim((0, 1.01e6))
    ax.legend()
    plt.tight_layout()
    plt.savefig('inventory_example.png')


def efficient_frontier_example():
    y = used_params['y']
    h = used_params['h']
    big_delta = used_params['Delta']
    big_t = used_params['T']
    sigma = used_params['sigma']
    epsilon = used_params['epsilon']
    eta = used_params['eta']
    gamma = used_params['gamma']

    n = 100
    lbdas = np.logspace(np.log10(1e2), np.log10(1e-2), n)
    means = np.zeros(n)
    variances = np.zeros(n)
    means_with_latency = np.zeros(n)
    variances_with_latency = np.zeros(n)

    for i in range(n):
        means_with_latency[i] = strategy_mean(y, h, big_delta, big_t, sigma,
                                              lbdas[i], epsilon, eta, gamma)
        variances_with_latency[i] = strategy_var(y, h, big_delta, big_t,
                                                 sigma, lbdas[i], epsilon,
                                                 eta, gamma)
        means[i] = strategy_mean(y, h, h, big_t, sigma, lbdas[i], epsilon,
                                 eta, gamma)
        variances[i] = strategy_var(y, h, h, big_t, sigma, lbdas[i], epsilon,
                                    eta, gamma)

    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)
    ax.plot(variances_with_latency, means_with_latency,
            label=r'$\Delta$=200ms')
    ax.plot(variances, means, label=r'$\Delta$=0ms')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Mean')
    ax.set_ylim((0, 3e6))
    ax.set_xlim((0, 3e6))
    ax.legend()
    plt.tight_layout()
    plt.savefig('efficient_frontier_example.png')

    # VaR example
    var99_with_latency = norm.ppf(.99)*np.sqrt(variances_with_latency)
    var99 = norm.ppf(.99)*np.sqrt(variances)

    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)
    ax.plot(var99_with_latency, means_with_latency,
            label=r'$\Delta$=200ms')
    ax.plot(var99, means, label=r'$\Delta$=0ms')
    ax.set_xlabel(r'$\mathcal{N}^{-1}(.99)\sqrt{\mathrm{Variance}}$')
    ax.set_ylabel('Mean')
    ax.set_ylim((0, 3e6))
    ax.set_xlim((0, 4e3))
    ax.legend()
    plt.tight_layout()
    plt.savefig('efficient_frontier_VaR_example.png')


def cost_of_latency_example():
    y = used_params['y']
    h = used_params['h']
    big_t = used_params['T']
    sigma = used_params['sigma']
    epsilon = used_params['epsilon']
    eta = used_params['eta']
    gamma = used_params['gamma']
    lbda = used_params['lbda']

    n = 500
    latencies = np.linspace(h, 0.5*big_t, n)
    cost_without_latency = strategy_mean(y, h, h, big_t, sigma, lbda, epsilon,
                                         eta, gamma)
    costs = np.zeros(n)

    for i in range(n):
        costs[i] = strategy_mean(y, h, latencies[i], big_t, sigma, lbda,
                                 epsilon, eta, gamma) - cost_without_latency

    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)
    ax.plot(days_to_seconds(latencies), costs)
    ax.set_xlabel('Latency')
    ax.set_ylabel('Expected cost')
    plt.tight_layout()
    plt.savefig('cost_of_latency_example.png')
