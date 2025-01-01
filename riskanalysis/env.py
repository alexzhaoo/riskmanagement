
import numpy as np
import pandas as pd
import gym
from gym import spaces
import PortfolioAnalysis as pa
class PortfolioAgentEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, window_size = 8):
        super(PortfolioAgentEnv, self).__init__()
        self.data = data
        self.initial_cash = initial_balance
        self.window_size = window_size
        self.num_stocks = len(data['Ticker'].unique())

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks * 17,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio = {ticker: self.initial_cash / self.num_stocks for ticker in self.data['Ticker'].unique()}
        self.weights = np.array([1 / self.num_stocks] * self.num_stocks)
        return self._get_obs()
    
    def _get_obs(self):
        grouped = self.data.groupby("Ticker")
        current_data = grouped.apply(lambda group: group.iloc[self.current_step:self.current_step + self.window_size])
        current_data = current_data.reset_index(drop=True)
        returns = current_data['Returns'].values
        volume = current_data['Volume'].values
        return np.concatenate([returns, volume ,self.weights])

    def step(self, action):
        
        
        grouped = self.data.groupby("Ticker")

        if self.current_step + self.window_size > len(self.data):
            return self.get_obs(), 0, True, {}
        
        current_data = grouped.apply(lambda group: group.iloc[self.current_step:self.current_step + self.window_size])
        current_data = current_data.reset_index(drop=True)
        returns = current_data['Returns']


        portfolio_value = sum([self.portfolio[ticker] * (1 + ret) for ticker, ret in zip(self.portfolio.keys(), returns)])

        self.portfolio = {ticker: portfolio_value * weight for ticker, weight in zip(self.portfolio.keys(), action)} # optional lines
        
        self.weights = action

        

        pivoted = current_data.pivot(index='Date', columns='Ticker', values='Returns')

        portfolio_return = np.dot(pivoted.sum(axis=0).values, action)

        
        portfolio_volatility = pa.diversification(returns, pivoted, action)
        # print(f"portfolio volatility: {portfolio_volatility}, portfolio return: {portfolio_return}, action: {action}")
        reward = portfolio_return / (portfolio_volatility + 1)

 
        self.current_step += self.window_size
        done = self.current_step >= len(self.data) - self.window_size

        return self._get_obs(), reward, done, {}





