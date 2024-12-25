import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def hist_var(data, alpha=0.95):
    var = np.percentile(data, 100 * (1 - alpha))
    return var

def parametric_var( std_dev, alpha = 0.95, time_period = 1):
    zscore = stats.norm.ppf(1 - alpha)
    var = std_dev * zscore * np.sqrt(time_period)
    return var

def hist_cvar(data, var):
    cvar = data[data <= var].mean()
    return cvar


def monte_carlo_var(data, alpha = 0.95, time_period = 1, num_trials = 10000):
    mean = np.mean(data)
    std_dev = np.std(data)
    rng = np.random.default_rng(42)
    simulated = rng.normal(mean, std_dev, num_trials)
    var = np.percentile(simulated, 100 * (1 - alpha)) * np.sqrt(time_period)
    return var

def diversification(data, weights):
    port_return = np.dot(data.mean(), weights)
    port_volatility = np.sqrt(weights.T @ (data.cov() @ weights))
    return port_return, port_volatility

def stress_test(data, stress_scenario):
    stressed_return = data.apply(lambda x: x*(1+stress_scenario))
    return stressed_return

def scenario_analysis(data, time):
    return data.loc[time]
                    
def extreme_value_theory(data, block_size = 10):
    block_maxima = data.groupby(np.arange(len(data))//block_size).max()
    params = stats.genextreme.fit(block_maxima)
    return params

def fit_fat_trailed(data):
    params = stats.t.fit(data)
    return params

if __name__ == '__main__':

    data = pd.read_csv("data/stock_data.csv")
    returnandticker = data[['Ticker', 'Returns']]
    returns = data['Returns']


    weights = np.array([1 / returns.shape[0]] * returns.shape[0]) # equal weights of each stock

    hist_var = hist_var(returns)
    parametric_var = parametric_var(np.std(returns))
    hist_cvar = hist_cvar(returns, hist_var)
    monte_carlo_var = monte_carlo_var(returns)

    print(f'Historical VaR: {hist_var}')
    print(f'Parametric VaR: {parametric_var}')
    print(f'Historical CVaR: {hist_cvar}')
    print(f'Monte Carlo VaR: {monte_carlo_var}')


    stressed_return = stress_test(returns, -0.1)

    pivoted_returns = data.pivot(index='date', columns='Ticker', values='Returns')
    port_return, port_volatility = diversification(returns, weights)

    print("portfolio return: ", port_return)
    print("portfolio volatility: ", port_volatility)

    evtparm = extreme_value_theory(returns.sum(axis=1))
    print("Extreme Value Theory: ", evtparm)

    fat_tail_params = fit_fat_trailed(returns)
    print("Fat Tail Parameters: ", fat_tail_params)

    
