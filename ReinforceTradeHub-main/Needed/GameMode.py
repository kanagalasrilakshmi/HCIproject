# initialize a past date
# randomly initialize a bucket
# loop 15 times in each - total 30 loops
# add a questionaire for each component / bucket after 15 loops

# one bucket lets the user view, create, manage portfolio, LLM explanabilty
# other bucket lets user view, create, get recommendations and create portfolio accordingly
# record the results in an excel - questionnare results/ survey results, 15 day portfolio results
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import json
import hashlib
from Sample import DQN,input_dim,output_dim,dow_30_list
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
import yfinance as yf


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

# write a function for getting stock price of a stock on a given date
# change this for current date
def fetch_latest_prices(stock_list,date):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    prices = {}
    for stock in stock_list:
        if stock != "UTX":
            ticker = yf.Ticker(stock)
            hist = ticker.history(start=date.strftime("%Y-%m-%d"), end=(date + timedelta(days=1)).strftime("%Y-%m-%d"))
            if not hist.empty:
                # If data is available for the date, return the closing price
                latest_price = hist['Close'].iloc[0]
            else:
                hist = ticker.history(start=(date - timedelta(days=7)).strftime("%Y-%m-%d"), end=(date + timedelta(days=1)).strftime("%Y-%m-%d"))
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                else:
                    latest_price = 0
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

# Initialize session state variables if they don't already exist
if 'date' not in st.session_state:
    st.session_state['date'] = datetime.strptime("2024-02-01", '%Y-%m-%d')
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'bucket' not in st.session_state:
    st.session_state['bucket'] = np.random.choice([0, 1])

# Function to handle proceeding to the next day
def proceed():
    st.session_state['date'] += timedelta(days=1)
    st.session_state['count'] += 1
    st.session_state['form_completed'] = False
    if st.session_state['count'] == 15:
        #st.session_state['bucket'] = 1 if st.session_state['bucket'] == 0 else 0
        if st.session_state['bucket'] == 0:
            st.session_state['bucket'] = 1
            embed_code = '''<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScmDESKWrouF0PgMO56SpYXvteasG5DUpQeUpeaR6NqbXvHew/viewform?usp=pp_url" width="700" height="520" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>'''
            st.markdown(embed_code, unsafe_allow_html=True)
            if st.button('I have completed the form'):
                st.session_state['form_completed'] = True
                # st.session_state['bucket'] = 1
                st.success("Thank you for filling out the form. Now moving to the next part.")
                
  
        else:
            st.session_state['bucket'] = 0
            # add apprpriate google form
            embed_code = '''<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScmDESKWrouF0PgMO56SpYXvteasG5DUpQeUpeaR6NqbXvHew/viewform?usp=pp_url" width="700" height="520" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>'''
            
            st.session_state.form_completed = False
            if not st.session_state.form_completed:
                st.markdown(embed_code, unsafe_allow_html=True)
                if st.button('I have completed the form'):
                    st.session_state['form_completed'] = True
                    # st.session_state['bucket'] = 0
                    st.success("Thank you for filling out the form. Now moving to the next part.")

            
    if st.session_state['count'] == 30:
        st.session_state['game_over'] = True
        # Game over condition
        # add appropriate google form
        if st.session_state['bucket'] == 1:
            embed_code = '''<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScmDESKWrouF0PgMO56SpYXvteasG5DUpQeUpeaR6NqbXvHew/viewform?usp=pp_url" width="700" height="520" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>'''
        # # add appropriate google form
        elif st.session_state['bucket'] == 0:
            embed_code = '''<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScmDESKWrouF0PgMO56SpYXvteasG5DUpQeUpeaR6NqbXvHew/viewform?usp=pp_url" width="700" height="520" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>'''
        
        if 'final_form_completed' not in st.session_state:
            st.session_state.final_form_completed = False
        if not st.session_state.final_form_completed:
            st.markdown(embed_code, unsafe_allow_html=True)
            if st.button('I have completed the form'):
                st.session_state.final_form_completed = True
        if st.session_state.final_form_completed:
            st.success("Thank you for filling out the form. Thanks for attempting this game")
        
menu = ["Home", "Login", "SignUp"]
choice = st.sidebar.selectbox("Menu", menu)
users = load_user_data()
portfolio_action = None
username = ""
st.title("Stock Portfolio Game App")

# Add a button to proceed to the next day and call the proceed function on click
if not st.session_state.get('game_over', False):

    # Display different buttons based on the bucket
    if st.session_state['bucket'] == 0:
        if choice == "Home":
            if 'user' in st.session_state:
                del st.session_state['user']
            st.subheader(f"Entered into Manual Recommendation Bucket for date : {st.session_state['date'].strftime('%Y-%m-%d')}")
            st.info("If done with portfolio creation click on proceed button")
            st.info("Create portfolio if you have not created portfolio")
            st.info("Then login to your user id with correct password and do the trading")
            if st.button("Proceed to Next Day"):
                proceed()
        
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
                if st.button("Total Portfolio Value", key = "Total Portfolio"):
                    st.session_state['portfolio_action'] = "Total Portfolio Value"
                portfolio_action = st.session_state.get('portfolio_action')
            
                # display the balance allocated to the portfolio
                if portfolio_action == "View Portfolios":
                    st.subheader("Your Portfolios")
                    if 'portfolios' in users[username]:
                        if len(users[username]['portfolios']) == 0:
                            st.write("You have no portfolios")
                        for portfolio in users[username]['portfolios']:
                            st.write(f"Portfolio Name: {portfolio['name']}")
                            st.write(f"Balance allocated for the portfolio : {portfolio['balance']}")
                            for stock_item in portfolio['stocks']:
                                st.write(f"{stock_item['stock']}: {stock_item['quantity']}")
                    else:
                        st.write("You have no portfolios.")

                elif portfolio_action == "Add New Portfolio":
                    new_portfolio_name = st.text_input("Portfolio Name")
                    balance = st.number_input("Enter Balance to be allocated($)", min_value=0.0, value=1000.00, step=100.0, format="%.2f")
                    flag = True
                    if st.button("Create Portfolio"):
                        for i in range(0,len(users[username]['portfolios'])):
                            if new_portfolio_name in users[username]['portfolios'][i]['name']:
                                st.success(f"Portfolio '{new_portfolio_name}' already exists!")
                                flag = False
                                break
                        if(flag):
                            users[username]['portfolios'].append({"name": new_portfolio_name, "balance":balance,"stocks": []})
                            save_user_data(users)
                            st.success(f"Portfolio '{new_portfolio_name}' created.")

                elif portfolio_action == "Manage Portfolio":
                    if 'portfolios' in users[username] and users[username]['portfolios']:
                        portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                        selected_portfolio_name = st.selectbox("Select Portfolio", portfolio_names)
                        selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == selected_portfolio_name), None)

                        stock = st.selectbox("Select Stock", available_stocks)
                        quantity = st.number_input("Quantity", min_value=0.01, value=1.00, step=0.01, format="%.2f")
                        
                        start_date = st.date_input("Start date", max_value=st.session_state['date'],value = st.session_state['date'])
                        end_date = st.date_input("End date", max_value=st.session_state['date'],value = st.session_state['date'])
                        data = yf.download(stock, start=start_date, end=end_date)

                        added_balance = st.number_input("Enter Balance to be allocated($)", min_value=0.0, step=100.0, format="%.2f")

                        # ticker = yf.Ticker(stock) # add a function
                        tod_date = st.session_state['date'].strftime('%Y-%m-%d')
                        # hist_data = ticker.history(start=tod_date, end=tod_date)
                        current_price_stock = fetch_latest_prices(stock_list=dow_30_list,date = tod_date)[stock]

                        if st.button("Buy Stocks"):
                            # get price of the stock on the given date
                            stock_val = (quantity*current_price_stock)
                            if stock_val<=selected_portfolio['balance']:
                                new_balance = selected_portfolio['balance'] - stock_val
                                selected_portfolio['balance'] = new_balance
                                flag = True
                                for stock_item in selected_portfolio['stocks']:
                                    if stock_item['stock'] == stock:
                                        stock_item['quantity']+=quantity
                                        flag = False
                                        break
                                if flag:
                                    selected_portfolio['stocks'].append({"stock": stock, "quantity": quantity})
                                save_user_data(users)
                                st.success(f"Added {quantity} shares of {stock} to the portfolio '{selected_portfolio_name}'.")
                            else:
                                st.write("Not enough balance")

                        if st.button("Sell Stocks"):
                            flag = True
                            for stock_item in selected_portfolio['stocks']:
                                if stock_item['stock'] == stock:
                                    flag = False
                                    if stock_item['quantity'] >= quantity:
                                        stock_item['quantity'] -= quantity
                                        if(stock_item['quantity']<=0):
                                            selected_portfolio['stocks'].remove(stock_item)
                                        stock_val = (quantity*current_price_stock)
                                        new_balance = selected_portfolio['balance'] + stock_val
                                        selected_portfolio['balance'] = new_balance
                                        st.success(f"Sold {quantity} shares of {stock} from the portfolio '{selected_portfolio_name}'.")
                                    else:
                                        st.write("Not enough number of stocks")
                                    break
                            if flag:
                                st.write("The stock is not present in this portfolio to sell")
                            save_user_data(users)

                        if st.button("Generate chart and Analysis"):                            
                            if not data.empty:
                                # Create a candlestick chart
                                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                                    open=data['Open'],
                                                                    high=data['High'],
                                                                    low=data['Low'],
                                                                    close=data['Close'])])
                                # Set titles
                                fig.update_layout(title=f'{stock} Candlestick Chart',
                                                xaxis_title='Date',
                                                yaxis_title='Price')

                                # Display the chart
                                st.plotly_chart(fig)
                                # add LLM explanability component
                                openai.api_key = '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'
                                data['Candle_Type'] = data.apply(get_candle_type, axis=1)
                                patterns_text = ' '.join([f"On {index.date()}, there was a {row['Candle_Type']} candle." 
                                for index, row in data.iterrows()])
                                
                                prompt = f"Given the following patterns of {stock}'s performance, provide an analysis and potential reasons for the trends:\n\n{patterns_text}"
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
                        
                        if st.button("Get value of the portfolio"):
                            value = selected_portfolio["balance"]
                            for stock_item in selected_portfolio['stocks']:
                                current_price_stock = fetch_latest_prices(stock_list=dow_30_list,date = tod_date)[stock_item['stock']]
                                stock_val = (stock_item['quantity']*current_price_stock)
                                value += stock_val
                            st.write(f"Total value of the user Portfolios is {value}")
                        
                        if st.button("Add balance"):
                            selected_portfolio["balance"]+=added_balance
                            save_user_data(users)
                            st.write("Balance is added")

                elif portfolio_action == "Total Portfolio Value":
                    if 'portfolios' in users[username] and users[username]['portfolios']:
                        portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                        total_sum = 0
                        for portfolio_name in portfolio_names:
                            selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == portfolio_name), None)
                            value = selected_portfolio["balance"]
                            for stock_item in selected_portfolio['stocks']:
                                # ticker = yf.Ticker(stock_item['stock'])
                                tod_date = st.session_state['date'].strftime('%Y-%m-%d')
                                # hist_data = ticker.history(start=tod_date, end=tod_date)
                                current_price_stock = fetch_latest_prices(stock_list=dow_30_list,date = tod_date)[stock_item['stock']] # add a function
                                stock_val = (stock_item['quantity']*current_price_stock)
                                value += stock_val
                            total_sum += value
                        st.write(f"Total value of the user Portfolios is {total_sum}")
                    else:
                        st.write("Please create a portfolio first.")

            
    elif st.session_state['bucket'] == 1:
        if choice == "Home":
            if 'user' in st.session_state:
                del st.session_state['user']
            st.subheader(f"Entered into Recommendation Bucket for date : {st.session_state['date'].strftime('%Y-%m-%d')}")
            st.info("If done with portfolio creation click on proceed button")
            st.info("Create portfolio if you have not created portfolio")
            st.info("The login to your user id with correct password and do the trading with our recommendation tool")
            if st.button("Proceed to Next Day"):
                proceed()

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
                if st.button("Recommend Portfolio", key="recommend_portfolio"):
                    st.session_state['portfolio_action'] = "Recommend Portfolio"
                if st.button("Total Portfolio Value", key = "Total Portfolio"):
                    st.session_state['portfolio_action'] = "Total Portfolio Value"
                if st.button("Manage Portfolio", key = "Manage Portfolio"):
                    st.session_state['portfolio_action'] = "Manage Portfolio"
                portfolio_action = st.session_state.get('portfolio_action')
            
                if portfolio_action == "View Portfolios":
                    st.subheader("Your Portfolios")
                    if 'portfolios' in users[username]:
                        if len(users[username]['portfolios']) == 0:
                            st.write("You have no portfolios")
                        for portfolio in users[username]['portfolios']:
                            st.write(f"Portfolio Name: {portfolio['name']}")
                            st.write(f"Balance allocated for the portfolio : {portfolio['balance']}")
                            for stock_item in portfolio['stocks']:
                                st.write(f"{stock_item['stock']}: {stock_item['quantity']}")
                    else:
                        st.write("You have no portfolios.")
            
                
                elif portfolio_action == "Manage Portfolio":
                    if 'portfolios' in users[username] and users[username]['portfolios']:
                        portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                        selected_portfolio_name = st.selectbox("Select Portfolio", portfolio_names)
                        selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == selected_portfolio_name), None)
                        added_balance = st.number_input("Enter Balance to be allocated($)", min_value=0.0, step=100.0, format="%.2f")
                        tod_date = st.session_state['date'].strftime('%Y-%m-%d')
                        
                        if st.button("Get value of the portfolio"):
                            value = selected_portfolio["balance"]
                            for stock_item in selected_portfolio['stocks']:
                                current_price_stock = fetch_latest_prices(stock_list=dow_30_list,date = tod_date)[stock_item['stock']]
                                stock_val = (stock_item['quantity']*current_price_stock)
                                value += stock_val
                            st.write(f"Total value of the user Portfolios is {value}")
                        
                        if st.button("Add balance"):
                            selected_portfolio["balance"]+=added_balance
                            save_user_data(users)
                            st.write("Balance is added")


                
                elif portfolio_action == "Add New Portfolio":
                    new_portfolio_name = st.text_input("Portfolio Name")
                    balance = st.number_input("Enter Balance to be allocated($)", min_value=0.0, value=1000.00, step=100.0, format="%.2f")
                    flag = True
                    if st.button("Create Portfolio"):
                        for i in range(0,len(users[username]['portfolios'])):
                            if new_portfolio_name in users[username]['portfolios'][i]['name']:
                                st.success(f"Portfolio '{new_portfolio_name}' already exists!")
                                flag = False
                                break
                        if(flag):
                            users[username]['portfolios'].append({"name": new_portfolio_name, "balance":balance,"stocks": []})
                            save_user_data(users)
                            st.success(f"Portfolio '{new_portfolio_name}' created.")

                elif portfolio_action == "Total Portfolio Value":
                    if 'portfolios' in users[username] and users[username]['portfolios']:
                        portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                        total_sum = 0
                        for portfolio_name in portfolio_names:
                            selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == portfolio_name), None)
                            value = selected_portfolio["balance"]
                            for stock_item in selected_portfolio['stocks']:
                                # ticker = yf.Ticker(stock_item['stock'])
                                tod_date = st.session_state['date'].strftime('%Y-%m-%d')
                                # hist_data = ticker.history(start=tod_date, end=tod_date)
                                current_price_stock = fetch_latest_prices(stock_list=dow_30_list,date = tod_date)[stock_item['stock']] # add a function
                                stock_val = (stock_item['quantity']*current_price_stock)
                                value += stock_val
                            total_sum += value
                        st.write(f"Total value of the user Portfolios is {total_sum}")
                    else:
                        st.write("Please create a portfolio first.")

                elif portfolio_action == "Recommend Portfolio":
                    if 'portfolios' in users[username] and users[username]['portfolios']:
                        portfolio_names = [portfolio['name'] for portfolio in users[username]['portfolios']]
                        selected_portfolio_name = st.selectbox("Select Portfolio for Recommendations", portfolio_names)
                        selected_portfolio = next((portfolio for portfolio in users[username]['portfolios'] if portfolio['name'] == selected_portfolio_name), None)
                        # enter the balance
                        #balance = st.number_input("Enter Available Balance ($)", min_value=0.0, value=1000.00, step=100.0, format="%.2f")
                        balance = selected_portfolio['balance']
                        holdings = []
                        current_prices = []
                        tod_date = st.session_state['date'].strftime('%Y-%m-%d')
                        latest_prices = fetch_latest_prices(dow_30_list,date=tod_date)
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
                            # empty portfolio
                            # we need to still give suggestions to the user
                            for stock in dow_30_list:
                                if stock!='UTX':
                                    holdings.append(0)
                                    current_prices.append(latest_prices[stock])
                            #st.write("No stocks in this portfolio.")
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
                                    # add the orders to the portfolio
                            if len(selling_orders)==0:
                                st.write(f"No need to sell the stocks")
                            if len(buying_orders)>0:
                                for orders in buying_orders:
                                    st.write(f"Buy {orders[1]} stocks from {orders[0]} , you will need {orders[2]} dollars to buy it")
                                    
                                    # add orders to the portfolio
                            if len(buying_orders)==0:
                                st.write(f"No need to buy any stocks")
                            st.write("Total Selling Value of the portfolio:", total_selling)
                            st.write("Total Buying Value of the portfolio:", total_buying)
                            st.write("Updated Balance of the portfolio:", balance)
                            # update the json file
                            # adding buying orders
                            for order_buy in buying_orders:
                                flag = True
                                for stock_item in selected_portfolio['stocks']:
                                    if stock_item['stock'] == order_buy[0]:
                                        stock_item['quantity']+=order_buy[1]
                                        flag = False
                                        break
                                if flag:
                                    selected_portfolio['stocks'].append({"stock": order_buy[0], "quantity": float(order_buy[1])})
                            # adding selling orders
                            for order_sell in selling_orders:
                                for stock_item in selected_portfolio['stocks']:
                                    if stock_item['stock'] == order_sell[0]:
                                        stock_item['quantity']-=float(order_sell[1])
                                        if stock_item['quantity']<=0:
                                            selected_portfolio['stocks'].remove(stock_item)
                            # update balance
                            selected_portfolio['balance'] = float(balance)
                            save_user_data(users)
                            st.success("Portfolio updated with recommendations.")

                    else:
                        st.write("Please create a portfolio first.")
else:
    st.write("Thank you for participating. The game is over.")
