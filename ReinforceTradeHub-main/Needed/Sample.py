
import pandas as pd
import numpy as np

import yfinance as yf

import gym



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


import numpy as np
import gym

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



import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import random

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



from tqdm import tqdm
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
    # Training loop
    epsilon = epsilon_start
    for episode in tqdm(range(num_episodes)):


        state = env.reset()
        total_reward = 0

        while True:

            # Get Action
            action, exp = select_action(state, epsilon, model, env.action_space)

            # Get Next State and Reward
            next_state, reward, done, _ = env.step(action)

            # Push to Replay Buffer
            replay_buffer.push(state, action, next_state, reward, done)
            
            # Update Next State
            state = next_state
            total_reward += reward

            # print(env.current_step)

            # train DQN/Update Q Values
            train_dqn(env, model, target_model, optimizer, replay_buffer, batch_size, gamma)

            portfolio_value = np.sum(env.current_prices * env.holdings) + env.balance

            if done:
                break


        # Update target network every target_update_freq episodes
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if(portfolio_value > 10000):
            print("*****************************")
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Portfolio_Value: {portfolio_value}")

    # After training, use the trained model for inference


    # saving the trained model
    torch.save(model.state_dict(),'policy_model.pth')

    # saving the model to resume training where we stopped
    torch.save(target_model.state_dict(),'target_model.pth')


if __name__ == "__main__":
    main()