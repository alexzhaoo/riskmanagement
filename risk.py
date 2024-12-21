import numpy as np
import pandas as pd



data  = pd.read_csv('data/stock_data.csv')


mytickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA']

myportfolio = data[data['Ticker'].isin(mytickers)]

portfolio_returns = myportfolio.groupby('date')['Returns'].sum()



