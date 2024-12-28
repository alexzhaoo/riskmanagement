import numpy as np
import pandas as pd
import gym
from gym import spaces
import PortfolioAnalysis as pa
class PortfolioAgentEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(PortfolioAgentEnv, self).__init__()
        self.data = data
        self.initial_cash = initial_balance

        self.num_stocks = len(data['Ticker'].unique())

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks * 4,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio = {ticker: self.initial_cash / self.num_stocks for ticker in self.data['Ticker'].unique()}
        self.weights = np.array([1 / self.num_stocks] * self.num_stocks)
        return self._get_obs()
    
    def _get_obs(self):
        current_data = self.data.iloc[self.current_step:self.current_step + self.num_stocks]
        returns = current_data['Returns'].values
        volume = current_data['Volume'].values
        cumulative_return = current_data['Cumulative Return'].values
        return np.concatenate([returns, volume, cumulative_return, self.weights])

    def step(self, action):
        action = np.clip(action, 0, 1)
        if action.sum() == 0:
            action = np.ones_like(action) / len(action) 
        else:
            action /= action.sum()

        current_data = self.data.iloc[self.current_step:self.current_step + self.num_stocks]
        returns = current_data['Returns'].values



        portfolio_value = sum([self.portfolio[ticker] * (1 + ret) for ticker, ret in zip(self.portfolio.keys(), returns)])


        self.portfolio = {ticker: portfolio_value * weight for ticker, weight in zip(self.portfolio.keys(), action)}
        self.weights = action


        portfolio_return = np.dot(returns, action)
        pivoted = current_data.pivot(index='Date', columns='Ticker', values='Returns')
        portfolio_volatility = pa.diversification(returns, pivoted, action)

        reward = portfolio_return /      (portfolio_volatility + 1e-6)

        self.current_step += self.num_stocks
        done = self.current_step >= len(self.data) - self.num_stocks

        return self._get_obs(), reward, done, {}


