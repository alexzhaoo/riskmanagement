
import yfinance as yf
import pandas as pd
import numpy as np


tickers = ['WBA', 'NVDA', 'PARA', 'MNST' , 'T']


stock_data = {}
for ticker in tickers:
    data = yf.download(ticker, start="2000-01-01", end="2023-12-31")
    data.reset_index(inplace=True)
    data.columns = data.columns.droplevel(0)  # Drop the multi-index level if present
    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']  # Explicitly rename
    data['Ticker'] = ticker
    stock_data[ticker] = data



combined_data = pd.concat(stock_data.values())

tickers_str = '_'.join(tickers)


combined_data['Returns'] = combined_data.groupby('Ticker')["Close"].pct_change()

combined_data.dropna(inplace=True)



latest_ipo_date = combined_data.groupby('Ticker')['Date'].min().max()
filtered_data = combined_data[combined_data['Date'] >= latest_ipo_date]


filtered_data = filtered_data[filtered_data['Returns'].abs() < 1]


filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])


filtered_data.set_index('Date', inplace=True)

def compound_returns(returns):
    return np.prod(1 + returns) - 1

monthly_data = filtered_data.groupby('Ticker').resample('M').agg({
    'Returns': compound_returns,  
    'Close': 'last',
    'Volume': 'sum'
}).reset_index()


monthly_data['Cumulative Return'] = monthly_data.groupby('Ticker')['Returns'].transform(lambda x: (1 + x).cumprod() - 1)

monthly_data.set_index('Date', inplace=True)

monthly_data.to_csv(rf'data\{tickers_str}monthlycompounded.csv', index=True)