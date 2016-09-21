from scipy.stats import norm
import numpy as np
import pandas as pd
from almgren_chriss import figure2_params
from optimal_splitting_under_latency import (model_with_latency, strategy_mean,
                                             strategy_var)
from utils import (days_to_seconds, miliseconds_to_days, preformatted_plot,
                   save_static_plot)


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


def inventory_plot():
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

    ax = preformatted_plot('Time (seconds)', 'Size of the inventory')
    ax.step(days_to_seconds(trajectory.index.values), trajectory.values,
            where='post', label=r'$\Delta$=200ms')
    ax.step(days_to_seconds(trajectory_without_latency.index.values),
            trajectory_without_latency.values, where='post',
            label=r'$\Delta$=0ms')
    ax.set_ylim((0, 1.01e6))
    save_static_plot(ax, 'inventory_example.png')


def efficient_frontier_plots():
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

    ax = preformatted_plot('Variance', 'Mean')
    ax.plot(variances_with_latency, means_with_latency,
            label=r'$\Delta$=200ms')
    ax.plot(variances, means, label=r'$\Delta$=0ms')
    ax.set_ylim((0, 3e6))
    ax.set_xlim((0, 3e6))
    save_static_plot(ax, 'efficient_frontier_example.png')

    # VaR example
    var99_with_latency = norm.ppf(.99)*np.sqrt(variances_with_latency)
    var99 = norm.ppf(.99)*np.sqrt(variances)

    ax = preformatted_plot(r'$\mathcal{N}^{-1}(.99)\sqrt{\mathrm{Variance}}$',
                           'Mean')
    ax.plot(var99_with_latency, means_with_latency,
            label=r'$\Delta$=200ms')
    ax.plot(var99, means, label=r'$\Delta$=0ms')
    ax.set_ylim((0, 3e6))
    ax.set_xlim((0, 4e3))
    save_static_plot(ax, 'efficient_frontier_VaR_example.png')


def cost_of_latency_plot():
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

    ax = preformatted_plot('Latency', 'Expected cost')
    ax.plot(days_to_seconds(latencies), costs)
    save_static_plot(ax, 'cost_of_latency_example.png')


if __name__ == '__main__':
    print_used_params()
    inventory_plot()
    efficient_frontier_plots()
    cost_of_latency_plot()
