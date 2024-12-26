import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

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
    rng = np.random.default_rng(412452)
    simulated = rng.normal(mean, std_dev, num_trials)
    var = np.percentile(simulated, 100 * (1 - alpha)) * np.sqrt(time_period)
    return var , simulated

def diversification(data, pivot,  weights):
    port_volatility = np.sqrt(weights.T @ (pivot.cov() @ weights))
    return port_volatility

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

def visualize_montcarlo(simulated_returns, num_walks=100):
    plt.figure(figsize=(10, 6))
    for _ in range(num_walks):
        walk = np.cumsum(np.random.choice(simulated_returns, size=len(simulated_returns), replace=True))
        plt.plot(walk, linewidth=0.5, alpha=0.3)
    plt.title('Monte Carlo Simulation of Returns')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.show()


if __name__ == '__main__':

    data = pd.read_csv("data/monthlycompounded2010.csv")
    
    returns = data['Returns']

    hist_var = hist_var(returns)
    parametric_var = parametric_var(np.std(returns))
    hist_cvar = hist_cvar(returns, hist_var)
    monte_carlo_var , simulated = monte_carlo_var(returns)

    print(f'Historical VaR: {hist_var}')
    print(f'Parametric VaR: {parametric_var}')  
    print(f'Historical CVaR: {hist_cvar}')
    print(f'Monte Carlo VaR: {monte_carlo_var}')


    pivoted = data.pivot(index='Date', columns='Ticker', values='Returns')
    weights = np.array([1 / pivoted.shape[1]] * pivoted.shape[1])
    rng = np.random.default_rng(129038)
    weights = rng.random(pivoted.shape[1])
    weights /= weights.sum()
    print(weights)
    stressed_return = stress_test(returns, -0.1)


    port_volatility = diversification(returns, pivoted, weights) 
    print("portfolio volatility: ", port_volatility)

    evtparm = extreme_value_theory(pivoted.sum(axis=1))
    print("Extreme Value Theory: ", evtparm)

    fat_tail_params = fit_fat_trailed(pivoted.sum(axis=1))
    print("Fat Tail Parameters: ", fat_tail_params)

    visualize_montcarlo(simulated)
