
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
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

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


        if len(current_data)/self.num_stocks < self.window_size:

            rows_need = int(self.window_size - len(current_data)/self.num_stocks)

            padding = grouped.apply(lambda group: group.iloc[:rows_need])
            padding = padding.reset_index(drop=True)

            current_data = pd.concat([padding, current_data], axis=0).reset_index(drop=True)


        returns = current_data['Returns'].values
        volume = current_data['Volume'].values 
        return np.concatenate([returns, volume ,self.weights])

    def step(self, action):
        
        
        grouped = self.data.groupby("Ticker")
        
        current_data = grouped.apply(lambda group: group.iloc[self.current_step:self.current_step + self.window_size])
        
        if len(current_data)/self.num_stocks < self.window_size:

            rows_need = int(self.window_size - len(current_data)/self.num_stocks)

            padding = grouped.apply(lambda group: group.iloc[:rows_need])
            padding = padding.reset_index(drop=True)

            current_data = pd.concat([padding, current_data], axis=0).reset_index(drop=True)
        current_data = current_data.reset_index(drop=True)
        returns = current_data['Returns']

        ticker_returns = current_data.groupby('Ticker')['Returns'].sum()

        returns_vector = ticker_returns.values


        
        self.weights = action

        if len(returns_vector) != len(action):
            raise ValueError(f"Mismatch: returns_vector has length {len(returns_vector)}, but action has length {len(action)}.")

        pivoted = current_data.pivot(index='Date', columns='Ticker', values='Returns')

        portfolio_return = np.dot(returns_vector, action)

        
        portfolio_volatility = pa.diversification(returns, pivoted, action)

        risk_aversion = 1
        reward = portfolio_return - risk_aversion * portfolio_volatility**2



        self.portfolio = portfolio_return 
        self.current_step += self.window_size
        # print(f"portfolio volatility: {portfolio_volatility}, portfolio return: {portfolio_return}}")
        done = self.current_step >= len(current_data) - self.window_size

        return self._get_obs(), reward, done, {}





