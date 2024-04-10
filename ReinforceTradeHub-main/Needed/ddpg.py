
import csv
import pandas as pd
import numpy as np
import yfinance as yf # type: ignore
import gym # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import random
from tqdm import tqdm # type: ignore
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import namedtuple, deque



dow_30_list = ["MMM","AXP","AAPL","BA","CAT",
               "CVX","CSCO","KO","DIS","DD",
               "XOM","GE","GS","HD","IBM",
               "INTC","JNJ","JPM","MCD","MRK",
               "MSFT","NKE","PFE","PG","TRV",
               "UNH","UTX","VZ","V","WMT"]

start_date = "2009-01-01"
end_date = "2014-12-31"
interval = "1d"


hist_daily_data = {}
for i,stock in enumerate(dow_30_list):

    stock_ticker = yf.Ticker(stock)

    #%Y-%m-%d
    hist_daily_data[stock] = stock_ticker.history(start=start_date, end=end_date, interval=interval)

    print(i, stock, hist_daily_data[stock].shape[0])

del hist_daily_data["UTX"]




class CustomActionSpace(gym.Space):
    def __init__(self, low, high):
        assert len(low) == len(high), "low and high should have the same length"
        self._low = np.array(low, dtype=np.int64)
        self._high = np.array(high, dtype=np.int64)
        self._nvec = self._high - self._low + 1

    @property
    def shape(self):
        return (len(self._low),)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def nvec(self):
        return self._nvec

    def contains(self, x):
        return np.all(x >= self._low) and np.all(x <= self._high)

    def sample(self):
        return np.random.randint(self._low, self._high + 1)

    def __repr__(self):
        return "CustomActionSpace"

    def __eq__(self, other):
        return np.all(self._low == other.low) and np.all(self._high == other.high)


class StockTradingEnvironment(gym.Env):
    def __init__(self, hist_daily_data):
        super(StockTradingEnvironment, self).__init__()

        self.hist_daily_data = hist_daily_data
        self.stock_names = list(hist_daily_data.keys())
        self.num_stocks = len(self.stock_names)
        self.current_step = 0
        self.initial_balance = 10000  # Initial balance for the agent
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks)
        self.current_prices = np.zeros(self.num_stocks)
        
        # Define the action space with a custom range
        action_low = [-100] * self.num_stocks
        action_high = [100] * self.num_stocks
        self.action_space = CustomActionSpace(action_low, action_high)


        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.num_stocks * 2 + 1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks)
        self.current_prices = self._get_current_prices()
        return self._get_observation()

    def _get_current_prices(self):
        return np.array([self.hist_daily_data[name]["Close"].iloc[self.current_step] for name in self.stock_names])

    def _get_observation(self):
        return np.concatenate([self.current_prices, self.holdings, [self.balance]])

    def step(self, action):
        # Execute the action (Buying: k > 0, Selling: k < 0)
        action = np.clip(action, -100, 100)  # Clip action values to [-100, 100]
        
        # Calculate portfolio value at the previous state (s)
        portfolio_value_at_s = np.sum(self.current_prices * self.holdings) + self.balance

        # Selling
        selling_orders = []
        for i, sell_order in enumerate(action):
            if sell_order < 0 and self.holdings[i] > 0:
                shares_to_sell = min(-sell_order, self.holdings[i])
                selling_orders.append((i, shares_to_sell, self.current_prices[i] * shares_to_sell))

        # Buying
        buying_orders = []
        for i, buy_order in enumerate(action):
            if buy_order > 0:
                max_shares = int(self.balance / self.current_prices[i])
                shares_to_buy = min(buy_order, max_shares)
                buying_orders.append((i, shares_to_buy, self.current_prices[i] * shares_to_buy))


        total_selling = sum(order[2] for order in selling_orders)
        total_buying = sum(order[2] for order in buying_orders)

        if selling_orders and total_selling < total_buying:
            # If there are actual selling transactions and the total money obtained from selling
            # is less than the total cost of buying, do not execute any trades and return to the previous state
            
            # Move to the next time step
            self.current_step += 1
            # Check if we reached the end of the historical data
            done = self.current_step+1 >= len(next(iter(self.hist_daily_data.values())))

            return self._get_observation(), 0, done, {}

        # Execute selling orders
        for order in selling_orders:
            i, shares_to_sell, earnings = order
            self.holdings[i] -= shares_to_sell
            self.balance += earnings

        # Execute buying orders
        for order in buying_orders:
            i, shares_to_buy, cost = order
            self.holdings[i] += shares_to_buy
            self.balance -= cost

        # Move to the next time step
        self.current_step += 1

        # Check if we reached the end of the historical data
        done = self.current_step+1 >= len(next(iter(self.hist_daily_data.values())))
        

        # Get the new stock prices for the next step
        self.current_prices = self._get_current_prices()

        # Calculate portfolio value at the current state (s0)
        portfolio_value_at_s0 = np.sum(self.current_prices * self.holdings) + self.balance
        
        # Calculate reward as the change in portfolio value
        reward = portfolio_value_at_s0 - portfolio_value_at_s

        
        print(f"Initial Balance: ${portfolio_value_at_s}")

        print(f"Selling Orders: {len(selling_orders)}, ${total_selling}")
        print(f"Buying Orders: {len(buying_orders)}, ${total_buying}")

        
        print(f"Final Balance: ${self.balance}")

        print(f"Balance the next day: ${portfolio_value_at_s0}")

        print(f"Reward: ${reward}")

        # Return the new observation, reward, whether the episode is done, and additional information
        return self._get_observation(), reward, done, {}



# Create the stock trading environment
env = StockTradingEnvironment(hist_daily_data)

# Reset the environment to the initial state
state = env.reset()

# Perform some random actions for a few time steps
for _ in range(3):
    # action = env.action_space.sample()  # Replace with your RL agent's action
    action = np.random.randint(-1, 2, size=len(env.stock_names))
    print(f"\n\nAction: {action}\n\n")

    next_state, reward, done, _ = env.step(action)
    # print(f"\n\nAction: {action}, Reward: {reward}, Done: {done}, Portfolio: {next_obs[-1]}")


# Close the environment (optional)
env.close()


# Define the Actor Model
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Assumes actions are in [-1, 1]
        )
    
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(88, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output is the Q-value
        )
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.network(state_action)

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBufferDDPG:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        batches = Experience(*zip(*experiences))
        return batches
    
    def __len__(self):
        return len(self.buffer)

def update_target_network(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

def train_ddpg(env, state_dim, action_dim, episodes, batch_size=128, capacity=10000, gamma=0.99):
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim + action_dim, 1)  # Ensure critic input dim is state + action
    target_actor = Actor(state_dim, action_dim)
    target_critic = Critic(state_dim + action_dim, 1)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters())
    critic_optimizer = optim.Adam(critic.parameters())

    replay_buffer = ReplayBufferDDPG(capacity)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = actor(state_tensor).detach().numpy().squeeze()
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                
                # Convert lists to tensors and ensure correct shapes
                states = torch.FloatTensor(batch[0])
                actions = torch.FloatTensor(batch[1]).view(batch_size, -1)  # Ensure actions are correctly shaped
                rewards = torch.FloatTensor(batch[2])
                next_states = torch.FloatTensor(batch[3])
                dones = torch.FloatTensor(batch[4])

                # Critic update
                Q_vals = critic(states, actions)
                next_actions = target_actor(next_states)
                next_Q = target_critic(next_states, next_actions).detach()
                Q_prime = rewards + gamma * next_Q.squeeze(1) * (1 - dones)
                critic_loss = F.mse_loss(Q_vals.squeeze(1), Q_prime)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor update
                policy_loss = -critic(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()

                # Update Target Networks
                update_target_network(target_actor, actor)
                update_target_network(target_critic, critic)

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")
        
    # Optionally, save your models here using torch.save
    # torch.save(actor.state_dict(), 'actor.pth')
    # torch.save(critic.state_dict(), 'critic.pth')

def run_inference(actor, env, num_episodes=10):
    """
    Run inference using the trained actor model.

    Parameters:
    - actor: The trained Actor model.
    - env: The environment compatible with OpenAI Gym's API.
    - num_episodes: Number of episodes to run the inference for.
    """
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Convert state into the appropriate format for the model
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from the actor model
            action = actor(state_tensor).detach().numpy()
            # Note: Adjust the action if your environment expects a different format or type
            
            # Take action in the environment
            next_state, reward, done, _ = env.step(action[0])  # Assume env expects a numpy array
            
            # Accumulate reward and set state to the new state
            episode_reward += reward
            state = next_state
        
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")


# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define a named tuple for transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Define the replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        # Flatten the buffer before sampling
        transitions = random.sample([t for t in self.buffer if t is not None], batch_size)
        return tuple(map(list, zip(*transitions)))  # Convert to tuple of lists

    def __len__(self):
        return len(self.buffer)


# Training loop
def train_dqn(env, model, target_model, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    state_batch = torch.FloatTensor(batch[0])
    try:
        action_batch = torch.LongTensor(batch[1])
        
    except:
        action_batch_list = batch[1]

        print(len(action_batch_list))

        # Convert each array to a NumPy array
        action_batch_np = [np.array(arr) for arr in action_batch_list]

        # Convert the list of NumPy arrays to a tensor
        action_batch = torch.LongTensor(action_batch_np)


    next_state_batch = torch.FloatTensor(batch[2])
    reward_batch = torch.FloatTensor(batch[3])
    done_mask = torch.BoolTensor(batch[4])

    # Compute Q-values for the current state-action pairs
    q_values = model(state_batch)

    # Map action values from [-100, 100] to [0, 200]
    action_batch_mapped = action_batch + 100

    # Ensure action_batch has the same number of dimensions as q_values
    action_batch_mapped = action_batch_mapped.unsqueeze(1)

    # Gather the Q-values corresponding to the actions taken
    q_values = q_values.gather(1, action_batch_mapped.squeeze(1))

    # Compute target Q-values using the Bellman equation
    with torch.no_grad():
        next_q_values = target_model(next_state_batch)

        # Reduce the dimensionality of next_q_values to match action space dimensionality
        next_q_values = next_q_values.view(next_q_values.size(0), env.action_space.shape[0], -1)

        # Get the maximum Q-values for the next state
        max_next_q_values = next_q_values.max(dim=2)[0]

        # Compute target Q-values using the Bellman equation
        target_q_values = reward_batch.unsqueeze(1) + (gamma * max_next_q_values * (~done_mask.unsqueeze(1)))


    # Compute MSE loss between predicted Q-values and target Q-values
    loss = F.mse_loss(q_values, target_q_values)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

# Epsilon-greedy policy
def select_action(state, epsilon, model, action_space):
    # Convert state to a PyTorch tensor and reshape it
    state_tensor = torch.FloatTensor(state).view(1, -1)
    
    # Choose action based on epsilon-greedy policy
    if np.random.rand() < epsilon:
        return action_space.sample(), "exploration"
    else:
        with torch.no_grad():
            q_values = model(state_tensor)
            # Reshape q_values to match the shape of the action space
            q_values = q_values.view(q_values.size(0), action_space.shape[0], -1)

            # Find the index of the maximum Q-value
            max_indices = q_values.argmax(dim=2)

            max_indices = max_indices.view(1, -1).numpy()[0]

            max_indices = max_indices - 100

            return max_indices, "exploitation"

env = StockTradingEnvironment(hist_daily_data)

# Define parameters
input_dim = env.observation_space.shape[0]
print(input_dim)
output_dim = sum(env.action_space.nvec)
print(output_dim)

model_path = 'ReinforceTradeHub-main/Needed/policy_model.pth'
target_model_path = 'ReinforceTradeHub-main/Needed/target_model.pth'


def main_ddpg():
    # Hyperparameters
    episodes = 500
    batch_size = 128
    capacity = 100000
    gamma = 0.99

    # Environment setup
    env = StockTradingEnvironment(hist_daily_data)
    state_dim = env.observation_space.shape[0]
    action_dim = len(env.action_space.low)  # Assuming action space is symmetric
    # Train DDPG
    train_ddpg(env, state_dim, action_dim, episodes, batch_size, capacity, gamma)


def main():
    # Create the stock trading environment
    
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 10
    batch_size = 64
    replay_buffer_capacity = 10000
    learning_rate = 0.001
    num_episodes = 1000

    # Initialize DQN models
    model = DQN(input_dim, output_dim)
    target_model = DQN(input_dim, output_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    if os.path.isfile(model_path) and os.path.isfile(target_model_path):
        print("Loading existing models...")
        model.load_state_dict(torch.load(model_path))
        target_model.load_state_dict(torch.load(target_model_path))
    else:
        print("Training new models...")
        # Training loop
        epsilon = epsilon_start
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            total_reward = 0

            while True:
                action, exp = select_action(state, epsilon, model, env.action_space)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, next_state, reward, done)
                state = next_state
                total_reward += reward

                train_dqn(env, model, target_model, optimizer, replay_buffer, batch_size, gamma)

                portfolio_value = np.sum(env.current_prices * env.holdings) + env.balance

                if done:
                    break

            if episode % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if(portfolio_value > 10000):
                print("*****************************")
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Portfolio_Value: {portfolio_value}")

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        torch.save(target_model.state_dict(), target_model_path)
    
    # Parameters for backtesting
    start_date_str = "2010-01-01"  
    end_date_str = "2010-12-31"  
    initial_balance = 10000  
    stock_list = ["MMM","AXP","AAPL","BA","CAT",
               "CVX","CSCO","KO","DIS","DD",
               "XOM","GE","GS","HD","IBM",
               "INTC","JNJ","JPM","MCD","MRK",
               "MSFT","NKE","PFE","PG","TRV", "UTX",
               "UNH","VZ","V","WMT"] 
    
    
    # Run backtesting
    model.eval()
    portfolio_values = run_backtest(model, stock_list, start_date_str, end_date_str)

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def fetch_stock_data(stock_list, start_date, end_date):
    """
    Fetches adjusted close prices for the given list of stocks.
    """
    data = yf.download(stock_list, start=start_date, end=end_date)
    return data['Close']

def execute_buy(balance, current_prices, quantity, price):
    cost = quantity * price
    if balance >= cost:
        balance -= cost
        return balance, quantity
    else:
        # Not enough balance to buy the intended quantity
        buyable_quantity = balance // price
        balance -= buyable_quantity * price
        return balance, buyable_quantity

def execute_sell(balance, holdings, quantity, price):
    balance += quantity * price
    holdings -= quantity
    return balance, holdings

# Function to calculate the total initial portfolio value
def calculate_initial_portfolio_value(holdings, opening_prices, initial_balance):
    initial_holdings_value = np.sum(np.array(holdings) * np.array(opening_prices))
    total_initial_portfolio_value = initial_balance + initial_holdings_value
    return total_initial_portfolio_value

# Function to get the opening prices for a list of stocks on a specific date
def get_opening_prices(stock_list, date):
    try:
        data = yf.download(stock_list, start=date, end=date + timedelta(days=1))
        if not data.empty:
            opening_prices = data['Open'].iloc[0].values
            return opening_prices
        else:
            print("No data returned for the given date.")
            return None
    except Exception as e:
        print(f"Failed to download stock data: {e}")
        return None


def run_backtest(model, stock_list, start_date_str, end_date_str, initial_balance=10000):
        
    # Remove 'UTX' if present in the stock list
    stock_list = [stock for stock in stock_list if stock != "UTX"]
    num_stocks = len(stock_list)
    
    opening_prices = get_opening_prices(stock_list, datetime.strptime(start_date_str, '%Y-%m-%d'))

    # Fetch historical data
    historical_data = fetch_stock_data(stock_list, start_date_str, end_date_str)

    balance = initial_balance
    holdings = np.zeros(len(stock_list))
    num_non_zero_holdings = np.random.randint(8, 25) 
    non_zero_indices = np.random.choice(num_stocks, num_non_zero_holdings, replace=False)
    for index in non_zero_indices:
        holdings[index] = np.random.randint(1, 7)
    for i in range(num_stocks):
        print(f"Stock: {stock_list[i]}, Initial Holdings: {holdings[i]}")

    if opening_prices is not None:
        initial_portfolio_value = calculate_initial_portfolio_value(holdings, opening_prices, initial_balance)
        print(f"Total Initial Portfolio Value: ${initial_portfolio_value}")
    else:
        print("Skipping portfolio value calculation due to missing data.")

    portfolio_values = []
    total_buys = 0
    total_sells = 0
    transaction_history = []

    for date, prices in historical_data.iterrows():
        current_prices = prices.values
        state = np.concatenate([[balance], holdings, current_prices])

        # Convert state to tensor and get action from the model
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_scores = model(state_tensor)
        actions = action_scores.argmax(1).numpy()  # Simplified action selection

        # Execute actions: For simplicity, buy/sell a single unit or hold
        for i, action in enumerate(actions):
            if action == 2 and balance >= current_prices[i]:  # Buy
                holdings[i] += 1
                balance -= current_prices[i]
                total_buys += 1
            elif action == 0 and holdings[i] > 0:  # Sell
                holdings[i] -= 1
                balance += current_prices[i]
                total_sells += 1

            transaction_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'action': 'buy' if action == 2 else 'sell' if action == 0 else 'hold',
                'stock': stock_list[i],
                'price': current_prices[i],
                'quantity': 1 if action in [0, 2] else 0
            })

        
        final_holdings_value = np.sum(holdings * current_prices)
        final_balance = balance
        print(f"Final cash balance: ${final_balance}")
        print(f"Value of final stock holdings: ${final_holdings_value}")
        print(f"Total buys: {total_buys}")
        print(f"Total sells: {total_sells}")


        log_file_name = f"transaction_history.csv"
        with open(log_file_name, mode='w', newline='') as file:
            fieldnames = ['date', 'action', 'stock', 'price', 'quantity']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()

            for transaction in transaction_history:
                writer.writerow(transaction)
                
            print(f"Transaction history saved to {log_file_name}")
            transaction_history.clear()
        
        # Update portfolio value
        portfolio_value = balance + np.sum(holdings * current_prices)
        portfolio_values.append(portfolio_value)
    for i in range(num_stocks):
        print(f"Stock: {stock_list[i]}, Final Holdings: {holdings[i]}")

    return portfolio_values

def calculate_sharpe_ratio(returns, risk_free_rate):
    # Calculate the excess returns by subtracting the risk-free rate
    excess_returns = returns - risk_free_rate
    
    # Calculate the average of the excess returns
    avg_excess_return = np.mean(excess_returns)
    
    # Calculate the standard deviation of the excess returns
    std_dev_excess_return = np.std(excess_returns)
    
    # If the standard deviation is zero, return 0 to avoid division by zero
    if std_dev_excess_return == 0:
        return 0
    
    # Calculate and return the Sharpe ratio
    sharpe_ratio = avg_excess_return / std_dev_excess_return
    return sharpe_ratio

if __name__ == "__main__":
    main