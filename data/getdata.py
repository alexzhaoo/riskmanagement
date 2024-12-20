from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from dotenv import load_dotenv
import os


api_key = os.getenv('SECRET_KEY')
ts = TimeSeries(key=api_key, output_format='pandas')


tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN'] 
stock_data = {}

for ticker in tickers:
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full') 
    data['Ticker'] = ticker
    stock_data[ticker] = data

combined_data = pd.concat(stock_data.values())

combined_data.to_csv('stock_data.csv')
