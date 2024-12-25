from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from dotenv import load_dotenv
import os


# api_key = os.getenv('SECRET_KEY')
# ts = TimeSeries(key=api_key, output_format='pandas')


# tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA'] 
# stock_data = {}

# for ticker in tickers:
#     data, meta_data = ts.get_daily(symbol=ticker, outputsize='full') 
#     data['Ticker'] = ticker
#     stock_data[ticker] = data

# combined_data = pd.concat(stock_data.values())

combined_data = pd.read_csv('data\stock_data.csv')


combined_data['Returns'] = combined_data.groupby('Ticker')['4. close'].transform(lambda x: (x - x.shift(1)) / x.shift(1) * 100) * -1

combined_data['Returns'] = combined_data.groupby('Ticker')['Returns'].shift(-1)


combined_data['Volatility'] = combined_data.groupby('Ticker')['Returns'].rolling(10).std().reset_index(level=0, drop=True)

combined_data['Volume_Norm'] = combined_data.groupby('Ticker')['5. volume'].transform(lambda x: x / x.rolling(10).mean())

combined_data.dropna(inplace=True)

combined_data.to_csv('data\stock_data.csv', index = False)
