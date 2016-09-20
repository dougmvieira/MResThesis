from math import ceil
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


np.seterr('raise')

test_params = {'S0':      50,            # dollars per share
               'X':       10**6,         # shares
               'T':       5,             # days
               'N':       5,
               'sigma':   0.95,          # dollars per share per day sqrt
               'alpha':   0.02,          # dollars per share per day
               'epsilon': 0.0625,        # dollars per share
               'gamma':   2.5*10**(-7),  # dollars per share squared
               'eta':     2.5*10**(-6),  # dollars per day
               'lbda u':  10**(-6),      # per dollar
               'lbda nu': 1.645}

def print_used_params():
    decision_lag = miliseconds_to_days(10)
    used_params = [('Initial inventory', test_params['X']),
                   ('Time horizon', '1s'),
                   ('Volatility (daily)', test_params['sigma']),
                   (r'Temporary impact ($\epsilon$)', test_params['epsilon']),
                   (r'Temporary impact ($\eta/h$)', test_params['eta']),
                   ('Permanent impact', test_params['gamma']),
                   ('Decision lag', '10ms'),
                   ('Execution delay', '200ms'),
                   ('Risk aversion', '0.1')]
    pd.DataFrame(used_params, columns=['Parameter', 'Value']
                 ).to_latex('used_params.tex', escape=False)
                

def harmonic_sum(x, y):
    return 1/((1/x)+(1/y))

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

def almgren_chriss(size, steps, final_time, volatility, risk_aversion,
                   temp_impact, perma_impact):
    sigma = volatility
    lbda = risk_aversion
    eta = temp_impact
    gamma = perma_impact
    big_t = final_time
    big_n = steps

    tau = big_t/big_n
    t_j = np.linspace(0, big_t, big_n + 1)

    eta_tilde, kappa = derived_params(eta, gamma, tau, lbda, sigma)

    inventory_values = size*np.sinh(kappa*(big_t - t_j))/np.sinh(kappa*big_t)

    return pd.Series(inventory_values, t_j)

def almgren_chriss_example():
    size = test_params['X']
    steps = test_params['N']
    final_time = test_params['T']
    volatility = test_params['sigma']
    temp_impact = test_params['eta']
    perma_impact = test_params['gamma']
 
    trajectory = almgren_chriss(size, steps, final_time, volatility,
                                2e-6, temp_impact, perma_impact)
    trajectory.plot()

def model_with_latency(size, decision_lag, execution_delay, final_time,
                       volatility, risk_aversion, temp_impact, perma_impact):
    h = decision_lag
    big_delta = execution_delay
    sigma = volatility
    lbda = risk_aversion
    eta = temp_impact
    gamma = perma_impact
    big_t = final_time

    big_n = ceil((big_t-big_delta)/h)
    big_t_tilde = big_n*h - big_delta
    eta_tilde, kappa = derived_params(eta, gamma, h, lbda, sigma)

    impulse_exec_times = np.arange(big_n)*h + big_delta
    time_stamps = np.hstack((0, impulse_exec_times))

    almgren_chriss_big_y_1 = (size*np.sinh(kappa*big_t_tilde)
                              )/np.sinh(kappa*(big_t_tilde+h))
    if big_delta == h:
        big_y_1 = almgren_chriss_big_y_1
    else:
        adjust = (eta_tilde*size)/(h*lbda*(big_delta-h)*sigma**2)
        big_y_1 = harmonic_sum(almgren_chriss_big_y_1,
                               adjust)

    inventory_values = np.hstack((size, big_y_1,
                                  almgren_chriss(big_y_1, big_n - 1,
                                                 big_t_tilde, volatility,
                                                 risk_aversion, temp_impact,
                                                 perma_impact).values[1:]))

    return pd.Series(inventory_values, time_stamps)

def strategy_mean(size, decision_lag, execution_delay, final_time, volatility,
                  risk_aversion, fix_impact, temp_impact, perma_impact):
    h = decision_lag
    big_delta = execution_delay
    y = size
    sigma = volatility
    lbda = risk_aversion
    epsilon = fix_impact
    eta = temp_impact
    gamma = perma_impact
    big_t = final_time

    big_n = ceil((big_t-big_delta)/h)
    big_t_tilde = big_n*h - big_delta
    eta_tilde, kappa = derived_params(eta, gamma, h, lbda, sigma)
 
    almgren_chriss_big_y_1 = (size*np.sinh(kappa*big_t_tilde)
                              )/np.sinh(kappa*(big_t_tilde+h))
    if big_delta == h:
        big_y_1 = almgren_chriss_big_y_1
    else:
        adjust = (eta_tilde*size)/(h*lbda*(big_delta-h)*sigma**2)
        big_y_1 = harmonic_sum(almgren_chriss_big_y_1,
                               adjust)

    return (epsilon*y + gamma/2*y**2 + 2*eta_tilde/h*(y-big_y_1)**2
            + eta_tilde*big_y_1**2*(
                np.tanh(.5*kappa*h)*(h*np.sinh(2*kappa*big_t_tilde)
                                     + 2*big_t_tilde*np.sinh(kappa*h))
                /(np.sinh(kappa*big_t_tilde)**2*np.sinh(kappa*h))))

def strategy_var(size, decision_lag, execution_delay, final_time, volatility,
                 risk_aversion, fix_impact, temp_impact, perma_impact):
    h = decision_lag
    big_delta = execution_delay
    sigma = volatility
    lbda = risk_aversion
    eta = temp_impact
    gamma = perma_impact
    big_t = final_time

    big_n = ceil((big_t-big_delta)/h)
    big_t_tilde = big_n*h - big_delta
    eta_tilde, kappa = derived_params(eta, gamma, h, lbda, sigma)

    inv_ac_big_y_1 = (np.sinh(kappa*(big_t_tilde+h))
                      /np.sinh(kappa*big_t_tilde))
    adjust = (h*lbda*(big_delta-h)*sigma**2)/eta_tilde
    big_y_1 = size/(inv_ac_big_y_1 + adjust)

    return big_delta*sigma**2*big_y_1 + (.5*sigma**2*big_y_1**2*(
             h*np.sinh(kappa*big_t_tilde)*np.cosh(kappa*(big_t_tilde-h))
             - big_t_tilde*np.sinh(kappa*h))
             /(np.sinh(kappa*big_t_tilde)**2*np.sinh(kappa*h)))
 
def almgren_chriss_var(size, steps, final_time, volatility, risk_aversion,
                       fix_impact, temp_impact, perma_impact):
    sigma = volatility
    lbda = risk_aversion
    epsilon = fix_impact
    eta = temp_impact
    gamma = perma_impact
    big_t = final_time
    big_n = steps

    tau = big_t/big_n
    t_j = np.linspace(0, big_t, big_n + 1)

    eta_tilde, kappa = derived_params(eta, gamma, tau, lbda, sigma)

    return (.5*sigma**2*size**2*
            (tau*np.sinh(kappa*big_t)*np.cosh(kappa*(big_t-tau))
             -big_t*np.sinh(kappa*tau))
            /(np.sinh(kappa*big_t)**2*np.sinh(kappa*tau)))

def days_to_seconds(days):
    seconds_in_a_trading_day = (9*60 + 30)*60
    return days*seconds_in_a_trading_day

def miliseconds_to_days(ms):
    minutes_in_a_trading_day = 9*60 + 30
    miliseconds_in_a_minute = 60*1000
    return ms/(minutes_in_a_trading_day*miliseconds_in_a_minute)

def inventory_example():
    size = test_params['X']
    decision_lag = miliseconds_to_days(10)
    execution_delay = miliseconds_to_days(200)
    final_time = miliseconds_to_days(1000)
    volatility = test_params['sigma']
    temp_impact = test_params['eta']*decision_lag
    perma_impact = test_params['gamma']
    lbda = .1

    trajectory = model_with_latency(size, decision_lag, execution_delay,
                                    final_time, volatility, lbda, temp_impact,
                                    perma_impact)
    trajectory_without_latency = model_with_latency(size, decision_lag,
                                                    decision_lag,
                                                    final_time, volatility,
                                                    lbda, temp_impact,
                                                    perma_impact)

    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)
    ax.step(days_to_seconds(trajectory.index.values), trajectory.values,
            where='post', label=r'$\Delta$=200ms')
    ax.step(days_to_seconds(trajectory_without_latency.index.values),
            trajectory_without_latency.values, where='post',
            label=r'$\Delta$=0ms')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Size of the inventory')
    ax.set_ylim((0,1.01e6))
    ax.legend()
    plt.tight_layout()
    plt.savefig('inventory_example.png')

def efficient_frontier_example():
    size = test_params['X']
    decision_lag = miliseconds_to_days(10)
    execution_delay = miliseconds_to_days(200)
    final_time = miliseconds_to_days(1000)
    volatility = test_params['sigma']
    fix_impact = test_params['epsilon']
    temp_impact = test_params['eta']*decision_lag
    perma_impact = test_params['gamma']

    n = 100
    lbdas = np.logspace(np.log10(1e2), np.log10(1e-2), n)
    means = np.zeros(n)
    variances = np.zeros(n)
    means_with_latency = np.zeros(n)
    variances_with_latency = np.zeros(n)

    for i in range(n):
        means_with_latency[i] = strategy_mean(
                size, decision_lag, execution_delay, final_time, volatility,
                lbdas[i], fix_impact, temp_impact, perma_impact)
        variances_with_latency[i] = strategy_var(
                size, decision_lag, execution_delay, final_time, volatility,
                lbdas[i], fix_impact, temp_impact, perma_impact)
        means[i] = strategy_mean(size, decision_lag, decision_lag,
                                 final_time, volatility, lbdas[i],
                                 fix_impact, temp_impact, perma_impact)
        variances[i] = strategy_var(size, decision_lag, decision_lag,
                                    final_time, volatility, lbdas[i],
                                    fix_impact, temp_impact, perma_impact)

    meanvar_with_latency = pd.Series(means_with_latency, variances_with_latency,
                                     name=r'$\Delta$=200ms')
    meanvar = pd.Series(means, variances, name=r'$\Delta$=0ms')

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
    size = test_params['X']
    decision_lag = miliseconds_to_days(10)
    execution_delay = miliseconds_to_days(200)
    final_time = miliseconds_to_days(1000)
    volatility = test_params['sigma']
    fix_impact = test_params['epsilon']
    temp_impact = test_params['eta']*decision_lag
    perma_impact = test_params['gamma']
    risk_aversion = .1

    n = 500
    latencies = np.linspace(decision_lag, 0.5*final_time, n)
    cost_without_latency = strategy_mean(
            size, decision_lag, decision_lag, final_time, volatility,
            risk_aversion, fix_impact, temp_impact, perma_impact)
    costs = np.zeros(n)

    for i in range(n):
        costs[i] = strategy_mean(
            size, decision_lag, latencies[i], final_time, volatility,
            risk_aversion, fix_impact, temp_impact, perma_impact
            ) - cost_without_latency

    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)
    ax.plot(days_to_seconds(latencies), costs)
    ax.set_xlabel('Latency')
    ax.set_ylabel('Expected cost')
    plt.tight_layout()
    plt.savefig('cost_of_latency_example.png')

