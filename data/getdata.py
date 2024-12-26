# DEPRECATED - This file is no longer used. The data is now fetched from the yfinance in the backend.




# 
# 
# from alpha_vantage.timeseries import TimeSeries
# import pandas as pd
# from dotenv import load_dotenv
# import os
# import numpy as np


# # api_key = os.getenv('SECRET_KEY')
# # ts = TimeSeries(key=api_key, output_format='pandas')


# # tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA'] 
# # stock_data = {}

# # for ticker in tickers:
# #     data, meta_data = ts.get_daily(symbol=ticker, outputsize='full') 
# #     data['Ticker'] = ticker
# #     data.reset_index(inplace=True) 
# #     stock_data[ticker] = data

# # combined_data = pd.concat(stock_data.values())

# # print(combined_data.head()) 


# # combined_data['Returns'] = combined_data.groupby('Ticker')['4. close'].transform(lambda x: (x - x.shift(1)) / x.shift(1) * 100) * -1

# # combined_data['Returns'] = combined_data.groupby('Ticker')['Returns'].shift(-1)


# # combined_data['Volatility'] = combined_data.groupby('Ticker')['Returns'].rolling(10).std().reset_index(level=0, drop=True)

# # combined_data['Volume_Norm'] = combined_data.groupby('Ticker')['5. volume'].transform(lambda x: x / x.rolling(10).mean())

# # combined_data.dropna(inplace=True)

# # combined_data.to_csv('data\stock_data.csv', index = False)

# data = pd.read_csv("data/stock_data.csv")


# latest_ipo_date = data.groupby('Ticker')['date'].min().max()

# data['Returns'] = data['Returns'] / 100

# filtered_data = data[data['date'] >= latest_ipo_date]

# filtered_data = filtered_data[filtered_data['Returns'].abs() < 1]

# filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# filtered_data.set_index('date', inplace=True)


# def compound_returns(returns):
#     return np.prod(1 + returns) - 1

# monthly_data = filtered_data.groupby('Ticker').resample('M').agg({
#     'Returns': compound_returns,  # Use the custom function for compounding
#     '4. close': 'last',
#     '5. volume': 'sum'
# }).reset_index()

# monthly_data['Cumulative Return'] = monthly_data.groupby('Ticker')['Returns'].transform(lambda x: (1 + x).cumprod() - 1)

# print(monthly_data.head())

# monthly_data.to_csv('data/monthly_data.csv', index=False)
