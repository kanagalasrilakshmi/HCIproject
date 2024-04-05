# code front end

# let user create porfolio
# add stocks into the portfolio
# make an estimation using the model in the portfolio and give suggestions

# let the user create user id and let him login into his own
# also check for existing user ids

import streamlit as st
import json
import hashlib
from Needed.Sample import DQN,input_dim,output_dim,dow_30_list
import numpy as np
import gym
import torch
import openai
import yfinance as yf
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import random
import plotly.graph_objects as go
available_stocks = ["MMM","AXP","AAPL","BA","CAT",
               "CVX","CSCO","KO","DIS","DD",
               "XOM","GE","GS","HD","IBM",
               "INTC","JNJ","JPM","MCD","MRK",
               "MSFT","NKE","PFE","PG","TRV",
               "UNH","VZ","V","WMT"]
def get_candle_type(row):
    if row['Close'] > row['Open']:
        return 'bullish'
    elif row['Close'] < row['Open']:
        return 'bearish'
    else:
        return 'neutral'

def fetch_latest_prices(stock_list):
    prices = {}
    for stock in stock_list:
        if stock != "UTX":
            ticker = yf.Ticker(stock)
            hist = ticker.history(period="1d")
            # Fetch the last available close price
            latest_price = hist.iloc[-1]['Close']
            prices[stock] = latest_price
    return prices

# Function to create a secure hash of the password
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Function to check the password hash against the stored hash
def check_hashes(password, hashed_text):
    if password == hashed_text:
        return hashed_text
    return False

# Function to load and save user data
user_data_file = 'user_data.json'

def load_user_data():
    try:
        with open(user_data_file, 'r') as file:
            users = json.load(file)
    except FileNotFoundError:
        users = {}
    return users

def save_user_data(users):
    with open(user_data_file, 'w') as file:
        json.dump(users, file)

# User Login and Registration
def login_user(username, password, users):
    if username in users:
        # Retrieve the user's data
        user_data = users.get(username, {})
        if 'password' in user_data and check_hashes(password, user_data['password']):
            return True
    return False

def create_user(username, password, users):
    if username not in users:
        users[username] = {
            'password': password,
            'portfolios': []  # Initialize with an empty list
        }
        save_user_data(users)
        return True
    return False

def main():
    st.title("Stock Portfolio App")

    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    users = load_user_data()
    portfolio_action = None
    username = ""

    if choice == "Home":
        if 'user' in st.session_state:
            del st.session_state['user']
        
        st.subheader("Home")

    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password")
        
        if st.sidebar.button("Login"):
            user_data = users.get(username, {})
            if 'password' in user_data and user_data['password']==str(password):
                st.success(f"Logged In as {username}")
                st.session_state['user'] = username
                st.session_state['portfolio_action'] = None  
            else:
                st.warning("Incorrect Username/Password")

        if 'user' in st.session_state:
            st.write(f"Logged in as: {st.session_state['user']}")
            if st.button("View Portfolios",key="view_portfolios"):
                st.session_state['portfolio_action'] = "View Portfolios"
            if st.button("Add New Portfolio", key="add_new_portfolio"):
                st.session_state['portfolio_action'] = "Add New Portfolio"
            if st.button("Manage Portfolio", key="manage_portfolio"):
                st.session_state['portfolio_action'] = "Manage Portfolio" 
            if st.button("Recommend Portfolio", key="recommend_portfolio"):
                st.session_state['portfolio_action'] = "Recommend Portfolio"
            if st.button("Generate Charts for stocks in portfolio", key="Generate Charts for stocks in portfolio"):
                st.session_state['portfolio_action'] = "Generate Charts for stocks in portfolio"
            portfolio_action = st.session_state.get('portfolio_action')

            if portfolio_action == "View Portfolios":
                st.subheader("Your Portfolios")
                if 'portfolios' in users[username]:
                    if len(users[username]['portfolios']) == 0:
                        st.write("You have no portfolios")
                    for portfolio in users[username]['portfolios']:
                        st.write(f"Portfolio Name: {portfolio['name']}")
                        for stock_item in portfolio['stocks']:
                            st.write(f"{stock_item['stock']}: {stock_item['quantity']} shares")
                else:
                    st.write("You have no portfolios.")

            elif portfolio_action == "Add New Portfolio":
                new_portfolio_name = st.text_input("Portfolio Name")
                flag = True
                if st.button("Create Portfolio"):
                    for i in range(0,len(users[username]['portfolios'])):
                        if new_portfolio_name in users[username]['portfolios'][i]['name']:
                            st.success(f"Portfolio '{new_portfolio_name}' already exists!")
                            flag = False
                            break
                    if(flag):
                        users[username]['portfolios'].append({"name": new_portfolio_name, "stocks": []})
                        save_user_data(users)
                        st.success(f"Portfolio '{new_portfolio_name}' created.")

            elif portfolio_action == "Manage Portfolio":
                # buy stocks
                # sell existing stocks
                # display chart

                if 'portfolios' in users[username] and users[username]['portfolios']:
                    portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                    selected_portfolio_name = st.selectbox("Select Portfolio", portfolio_names)
                    selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == selected_portfolio_name), None)

                    stock = st.selectbox("Select Stock", available_stocks)
                    quantity = st.number_input("Quantity", min_value=0.01, value=1.00, step=0.01, format="%.2f")

                    if st.button("Add to Selected Portfolio"):
                        selected_portfolio['stocks'].append({"stock": stock, "quantity": quantity})
                        save_user_data(users)
                        st.success(f"Added {quantity} shares of {stock} to the portfolio '{selected_portfolio_name}'.")
                else:
                    st.write("Please create a portfolio first.")
            
            elif portfolio_action == "Recommend Portfolio":
                # proceed with the strategy planned by ai 
                if 'portfolios' in users[username] and users[username]['portfolios']:
                    portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                    selected_portfolio_name = st.selectbox("Select Portfolio for Recommendations", portfolio_names)
                    selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == selected_portfolio_name), None)
                    # enter the balance
                    balance = st.number_input("Enter Available Balance ($)", min_value=0.0, value=1000.00, step=100.0, format="%.2f")
                    holdings = []
                    current_prices = []
                    latest_prices = fetch_latest_prices(dow_30_list)
                    st.write(f"Stocks in '{selected_portfolio_name}' Portfolio:")
                    portfolio = {}
                    if selected_portfolio and 'stocks' in selected_portfolio and selected_portfolio['stocks']:
                        for stock_item in selected_portfolio['stocks']:
                            portfolio[stock_item['stock']] = stock_item['quantity']
                        for stock in dow_30_list:
                            if stock!='UTX':
                                holdings.append(portfolio.get(stock, 0))
                                current_prices.append(latest_prices[stock])
                    else:
                        st.write("No stocks in this portfolio.")
                    if st.button("Get Recommendations"):

                        input_vector = current_prices + holdings + [balance]
                        model = DQN(input_dim, output_dim)
                        model.load_state_dict(torch.load('policy_model.pth'))
                        model.eval()
                        state_tensor = torch.FloatTensor(input_vector).unsqueeze(0) 
                        with torch.no_grad():
                            action_predictions = model(state_tensor).squeeze(0)
                        num_stocks = len(dow_30_list)-1
                        max_actions = action_predictions.view(1,num_stocks, -1)
                        max_indices = max_actions.argmax(dim=2)
                        max_indices = max_indices.view(1, -1).numpy()[0]
                        max_indices = max_indices - 100
                        selling_orders = []
                        buying_orders = []

                        # Process selling orders
                        for i, action_value in enumerate(max_indices):
                            if action_value < 0:  # Sell action
                                shares_to_sell = min(-action_value, holdings[i])
                                if shares_to_sell > 0:  # Check if there are shares to sell
                                    selling_orders.append((available_stocks[i], shares_to_sell, current_prices[i] * shares_to_sell))
                                    balance += current_prices[i] * shares_to_sell  # Update balance
                                    holdings[i] -= shares_to_sell  # Update holdings

                        # Process buying orders
                        for i, action_value in enumerate(max_indices):
                            if action_value > 0:  # Buy action
                                max_shares_to_buy = balance // current_prices[i]  # Integer division to find maximum shares that can be bought
                                shares_to_buy = min(action_value, max_shares_to_buy)
                                if shares_to_buy > 0:  # Check if it's possible to buy any shares
                                    buying_orders.append((available_stocks[i], shares_to_buy, current_prices[i] * shares_to_buy))
                                    balance -= current_prices[i] * shares_to_buy  # Update balance
                                    holdings[i] += shares_to_buy  # Update holdings
                        
                        total_selling = sum(order[2] for order in selling_orders)
                        total_buying = sum(order[2] for order in buying_orders)
                        st.write("Recommendations:")
                        if len(selling_orders)>0:
                            for orders in selling_orders:
                                st.write(f"Sell {orders[1]} stocks from {orders[0]} , you will get {orders[2]} dollars")
                        if len(selling_orders)==0:
                            st.write(f"No need to sell the stocks")
                        if len(buying_orders)>0:
                            for orders in buying_orders:
                                st.write(f"Buy {orders[1]} stocks from {orders[0]} , you will need {orders[2]} dollars to buy it")
                        if len(buying_orders)==0:
                            st.write(f"No need to buy any stocks")
                        st.write("Total Selling Value of the portfolio:", total_selling)
                        st.write("Total Buying Value of the portfolio:", total_buying)
                        st.write("Updated Balance of the portfolio:", balance)
                else:
                    st.write("Please create a portfolio first.")

            elif portfolio_action == "Generate Charts for stocks in portfolio":
                # get portfolio name
                if 'portfolios' in users[username] and users[username]['portfolios']:
                    portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                    selected_portfolio_name = st.selectbox("Select Portfolio for Recommendations", portfolio_names)
                    selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == selected_portfolio_name), None)
                    if selected_portfolio and 'stocks' in selected_portfolio and selected_portfolio['stocks']:
                        stock_options = [stock_item['stock'] for stock_item in selected_portfolio['stocks']]
                        selected_stock = st.selectbox("Select a Stock for Recommendations", options=stock_options)
                        start_date = st.date_input("Start date")
                        end_date = st.date_input("End date")
                        data = yf.download(selected_stock, start=start_date, end=end_date)
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
                            # add LLM explanability component
                            openai.api_key = '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'
                            data['Candle_Type'] = data.apply(get_candle_type, axis=1)
                            patterns_text = ' '.join([f"On {index.date()}, there was a {row['Candle_Type']} candle." 
                              for index, row in data.iterrows()])
                            
                            prompt = f"Given the following patterns of {selected_stock}'s performance, provide an analysis and potential reasons for the trends:\n\n{patterns_text}"
                            messages = [{
                                "role": "system",
                                "content": "You are a helpful assistant."
                            }, {
                                "role": "user",
                                "content": prompt  # Your previously defined prompt or question
                            }]
                            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
                            st.write(response.choices[0].message['content'].strip())

                        else:
                            st.write("No data available for the selected date range.")
                    else:
                        st.write("No stocks in the portfolio.")
                else:
                    st.write("Please create a portfolio first.")


    elif choice == "SignUp":
        if 'user' in st.session_state:
            del st.session_state['user']
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("SignUp"):
            create_user_response = create_user(new_user, new_password, users)
            if create_user_response:
                st.success("You have successfully created an account")
                st.info("Go to Login Menu to login")
            else:
                st.warning("User already exists")

if __name__ == '__main__':
    main()