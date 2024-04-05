from Sample import DQN,input_dim,output_dim
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import random
# import yfinance as yf

# # Assuming portfolio is a dictionary with stock symbols as keys and the number of shares as values.
# portfolio = {
#     "AAPL": 10,
#     "MSFT": 5,
#     # Add other stocks and their quantities as needed.
# }

# # Function to get the latest prices for each stock in the portfolio.
# def get_latest_prices(portfolio):
#     prices = {}
#     for stock in portfolio:
#         ticker = yf.Ticker(stock)
#         hist = ticker.history(period="1d")
#         latest_price = hist['Close'].iloc[-1]  # Get the latest close price
#         prices[stock] = latest_price
#     return prices

# # Fetch the latest prices for the portfolio
# latest_prices = get_latest_prices(portfolio)
# print(latest_prices)

# # Function to create the state (model input) from the portfolio and the latest prices
# def create_state(portfolio, latest_prices):
#     # Initialize an array to hold the state
#     state = []
    
#     # Add holdings to the state
#     for stock in portfolio:
#         state.append(portfolio[stock])  # Append the number of shares held
    
#     # Add prices to the state
#     for stock in latest_prices:
#         state.append(latest_prices[stock])  # Append the latest stock price
    
#     # If your model expects additional features like indicators or a cash balance, add them to the state here.
    
#     return state

# # Create the model input (state) from the portfolio and latest prices
# model_input = create_state(portfolio, latest_prices)
# print(model_input)

# import torch
# from Sample import DQN,input_dim,output_dim  # Replace with your actual file and class names

# # Initialize the model and load the saved weights
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load('policy_model.pth'))
model.eval()




portfolio = {
    "AAPL": 10,
    "MSFT": 5,
}

dow_30_list = [
    "MMM", "AXP", "AAPL", "BA", "CAT",
    "CVX", "CSCO", "KO", "DIS", "DD",
    "XOM", "GE", "GS", "HD", "IBM",
    "INTC", "JNJ", "JPM", "MCD", "MRK",
    "MSFT", "NKE", "PFE", "PG", "TRV",
    "UNH", "VZ", "V", "WMT"
]

# Example current prices for each stock in the Dow 30 list
current_prices_example = {
    "MMM": 180.00, "AXP": 150.00, "AAPL": 130.00, "BA": 220.00, "CAT": 230.00,
    "CVX": 120.00, "CSCO": 55.00, "KO": 50.00, "DIS": 180.00, "DD": 75.00,
    "XOM": 90.00, "GE": 100.00, "GS": 350.00, "HD": 310.00, "IBM": 145.00,
    "INTC": 60.00, "JNJ": 170.00, "JPM": 155.00, "MCD": 240.00, "MRK": 85.00,
    "MSFT": 250.00, "NKE": 140.00, "PFE": 40.00, "PG": 130.00, "TRV": 150.00,
    "UNH": 400.00, "VZ": 58.00, "V": 210.00, "WMT": 140.00
}

# Assume balance is $3,000
balance = 30000
def prepare_input_vector(portfolio, dow_30_list, current_prices, balance):
    # Prepare current prices array
    current_prices_array = [current_prices[stock] for stock in dow_30_list]
    
    # Prepare holdings array
    holdings_array = [portfolio.get(stock, 0) for stock in dow_30_list]
    
    # Combine into a single input vector
    input_vector = current_prices_array + holdings_array + [balance]
    
    return input_vector

# Create the input vector
input_vector = prepare_input_vector(portfolio, dow_30_list, current_prices_example, balance)
print("Input Vector:", input_vector)
print("Length of Input Vector:", len(input_vector))

# Convert the model input to a tensor
state_tensor = torch.FloatTensor(input_vector).unsqueeze(0)  # Add a batch dimension (1, number_of_features)

# Get the model's action predictions
with torch.no_grad():
    action_predictions = model(state_tensor).squeeze(0)  # Remove batch dimension with squeeze()

# Print the predicted actions
num_stocks = len(dow_30_list)
print("Predicted actions:", action_predictions.numpy())
print(len(action_predictions.numpy()))
max_actions = action_predictions.view(1,num_stocks, -1)
max_indices = max_actions.argmax(dim=2)
max_indices = max_indices.view(1, -1).numpy()[0]

max_indices = max_indices - 100
print(max_indices)


holdings = np.array([portfolio.get(stock, 0) for stock in dow_30_list])  # holdings per stock
current_prices = np.array([current_prices_example[stock] for stock in dow_30_list])  # current price per stock
balance = 3000000  # Available cash balance

print(holdings)
print(current_prices)
print(max_indices)
selling_orders = []
buying_orders = []

# Process selling orders
for i, action_value in enumerate(max_indices):
    if action_value < 0:  # Sell action
        shares_to_sell = min(-action_value, holdings[i])
        if shares_to_sell > 0:  # Check if there are shares to sell
            selling_orders.append((dow_30_list[i], shares_to_sell, current_prices[i] * shares_to_sell))
            balance += current_prices[i] * shares_to_sell  # Update balance
            holdings[i] -= shares_to_sell  # Update holdings

# Process buying orders
for i, action_value in enumerate(max_indices):
    if action_value > 0:  # Buy action
        max_shares_to_buy = balance // current_prices[i]  # Integer division to find maximum shares that can be bought
        shares_to_buy = min(action_value, max_shares_to_buy)
        if shares_to_buy > 0:  # Check if it's possible to buy any shares
            buying_orders.append((dow_30_list[i], shares_to_buy, current_prices[i] * shares_to_buy))
            balance -= current_prices[i] * shares_to_buy  # Update balance
            holdings[i] += shares_to_buy  # Update holdings
print(buying_orders)
print(selling_orders)
print(holdings)
# Summarize total buying and selling
total_selling = sum(order[2] for order in selling_orders)
total_buying = sum(order[2] for order in buying_orders)

# Output actions
print("Selling Orders:", selling_orders)
print("Buying Orders:", buying_orders)
print("Total Selling Value:", total_selling)
print("Total Buying Value:", total_buying)
print("Updated Balance:", balance)