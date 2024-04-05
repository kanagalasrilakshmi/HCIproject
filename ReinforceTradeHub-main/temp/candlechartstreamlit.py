import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Define your portfolio as a list of stock symbols
portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]

# Let the user select a stock symbol from the portfolio
selected_stock = st.selectbox('Select a stock for analysis:', portfolio)

# Input for start and end date
start_date = st.date_input("Start date")
end_date = st.date_input("End date")

# Fetch the historical data for the selected stock
data = yf.download(selected_stock, start=start_date, end=end_date)

# Check if data is empty
if not data.empty:
    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    # Set titles
    fig.update_layout(title=f'{selected_stock} Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Price')

    # Display the chart
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected date range.")
