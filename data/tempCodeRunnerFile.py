combined_data = pd.read_csv(rf'data\{tickers_str}.csv')

# combined_data['Returns'] = combined_data.groupby('Ticker')["Close"].pct_change()

# combined_data.dropna(inplace=True)



# latest_ipo_date = combined_data.groupby('Ticker')['Date'].min().max()
# filtered_data = combined_data[combined_data['Date'] >= latest_ipo_date]


# filtered_data = filtered_data[filtered_data['Returns'].abs() < 1]


# filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])


# filtered_data.set_index('Date', inplace=True)


# filtered_data.to_csv(r'data\noncompoundeddailyreturns.csv', index = True)

# def compound_returns(returns):
#     return np.prod(1 + returns) - 1

# monthly_data = filtered_data.groupby('Ticker').resample('M').agg({
#     'Returns': compound_returns,  
#     'Close': 'last',
#     'Volume': 'sum'
# }).reset_index()


# monthly_data['Cumulative Return'] = monthly_data.groupby('Ticker')['Returns'].transform(lambda x: (1 + x).cumprod() - 1)

# monthly_data.set_index('Date', inplace=True)

# monthly_data.to_csv(r'data\monthlycompounded.csv', index=True)