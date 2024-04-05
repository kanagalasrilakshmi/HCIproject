import mplfinance as mpf
import yfinance as yf

# Define the ticker symbol for the stock you are interested in
ticker_symbol = 'AAPL'

# # Download historical data for this ticker
# data = yf.download(tickers=ticker_symbol, period='1mo', interval='1d')

# # Generate candlestick chart
# mpf.plot(data, type='candle', style='charles', title=f'{ticker_symbol} Candlestick Chart')


ticker = yf.Ticker(ticker_symbol)
tod_date = "2024-03-01"
end_date = "2024-03-02"
hist_data = ticker.history(start=tod_date, end=end_date)
current_price_stock = hist_data['Close'].iloc[0]
print(current_price_stock)
