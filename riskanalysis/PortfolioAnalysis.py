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

def calculate_var(data):
    returns = data['Returns']
    hist_var_value = hist_var(returns)
    parametric_var_value = parametric_var(np.std(returns))
    hist_cvar_value = hist_cvar(returns, hist_var_value)
    monte_carlo_var_value, simulated = monte_carlo_var(returns)

    print(f'Historical VaR: {hist_var_value}')
    print(f'Parametric VaR: {parametric_var_value}')  
    print(f'Historical CVaR: {hist_cvar_value}')
    print(f'Monte Carlo VaR: {monte_carlo_var_value}')









def main(portweights):
    data = pd.read_csv("data/WBA_NVD_PARA_MNST_Tmonthlycompounded.csv")

    agentweights = pd.read_csv(portweights)

    agentweights = agentweights.to_numpy().flatten()


    returns = data['Returns']
    hist_var_value, parametric_var_value, hist_cvar_value, monte_carlo_var_value = calculate_var(data)

    pivoted = data.pivot(index='Date', columns='Ticker', values='Returns')

    weights = np.array([1 / pivoted.shape[1]] * pivoted.shape[1])
    rng = np.random.default_rng(129038)
    randomweights = rng.random(pivoted.shape[1])
    randomweights /= randomweights.sum()

    evtparm = extreme_value_theory(pivoted.sum(axis=1))
    print("Extreme Value Theory: ", evtparm)

    fat_tail_params = fit_fat_trailed(pivoted.sum(axis=1))
    print("Fat Tail Parameters: ", fat_tail_params)

    port_volatility = diversification(returns, pivoted, randomweights) 
    print("Random Weights portfolio volatility: ", port_volatility)

    equal_portfolio_volatility = diversification(returns, pivoted, randomweights) 
    print("Equal WeightsPortfolio volatility: ", equal_portfolio_volatility)

    agent_volatility = diversification(returns, pivoted, agentweights) 
    print("Agent Portfolio volatility: ", agent_volatility)

    pivoted_cumulative = data.pivot(index='Date', columns='Ticker', values='Cumulative Return')
    
    portfolio_agent = (pivoted_cumulative @ agentweights).dropna()  
    portfolio_random = (pivoted_cumulative @ randomweights).dropna()
    portfolio_equal = (pivoted_cumulative @ weights).dropna() 

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_random.index, portfolio_random.values, label="Random Weights", color="blue")
    plt.plot(portfolio_equal.index, portfolio_equal.values, label="Equal Weights", color="green", linestyle="--")
    plt.plot(portfolio_agent.index, portfolio_agent.values, label="Agent Weights", color="red", linestyle=":")


    plt.title("Portfolio Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()