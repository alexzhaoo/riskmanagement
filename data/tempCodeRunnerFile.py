data['date'] = pd.to_datetime(data['date'])

# data.set_index('date', inplace=True)


# pivoted = data.pivot(index='date', columns='Ticker', values='Returns')

# yearly_returns = (1 + pivoted).resample('Y').prod() - 1
# print(yearly_returns)