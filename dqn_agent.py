from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Hyperparameters


# Gamma is how much the agent values future vs immediate rewards
# 0.99 means a reward 100 steps from now is 0.99^100 (37%) of immediate reward
# I chose a high gamma since episodes are quite long with 288 steps
GAMMA = 0.99

# Learning rate, 0.001 is standard
LR = 1e-3

# How many expereinces from the reply buffer I pull before a training step
BATCH_SIZE = 64

# Dont start training until the buffer has at least 1000 experiences
MIN_BUFFER_SIZE = 1_000

# Epsilon is the explore/exploit tradeoff conroller
# Start at 1 which is fully random exploration, and decay down to 0.05 by the end which is more exploit focused
# Never fully reach 0 since the environemnt is always changing and the agent needs adaptability
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / 500 # reach 0.05 after 700 episodes

# After the online network has done 500 gradient updates copy the wieghts to the target network
# This is needed since the target network needs to stay frozen long enough to give stable targets
TARGET_UPDATE_FREQ = 500


# ReplayBuffer

# Instead of training on transitions which are obviously correlated, put all expereinces into storage
# This will be randomly sampled for training to break the correlation between steps and give diverse targets
class ReplayBuffer:
    def __init__(self, capacity=50_000):
        # using deque with maxlen drops the oldest entry of the buffer when it is full
        # This keeps the most recent 50k transitions which is about 150 full episodes
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        # Store a transition
        # store "done" if it was the last step of an episode. No future rewards
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        # Randomly sample transitions of batch_size
        # zip(*batch) unzips the list of tuples into the separate lists for each field
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        # Convert to numpy arrays so TensorFlow can work
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# QNetwork

# Given an observation of the ED, output one Q-value for each of the 21 possible actions
# The Q-value represents how good it is to take the action right now


# 65 inputs -> 128 Dense -> 128 Dense -> 21 output neurons
# No activation on the output layer, this is standard for Q-values since they can be positive or negative
def build_qnetwork():
    return keras.Sequential([
        keras.layers.Input(shape=(65,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(21),  # one Q-value per action, no activation
    ])


# DQNAgent

# The agent holds two copies of the Q-network, online_net for the one we train, target_net as a frozen target
# Having two network is what makes the DQN work, without it the gradient update changes the targets themselves
# To keep targets stable keep a version frozen for 500 steps
class DQNAgent:
    def __init__(self):
        self.online_net = build_qnetwork()
        self.target_net = build_qnetwork()
        # make sure both networks start with same weights
        self.sync_target()

        # Adam optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=LR)
        # Decided to use Huber loss instead of MSE. Rwards can get pretty large over so many steps
        # Squaring these rewards in MSE create very large gradients
        # Huber loss is similar to MSE for small errors and MAE (Mean absolute error) for large ones
        self.huber_loss = keras.losses.Huber(delta=10.0)
        self.replay_buffer = ReplayBuffer()
        self.epsilon = EPSILON_START

        # Counts gradient updates to I know when to sync with the target network
        self.steps_done = 0

    def select_action(self, obs, epsilon=None):
        # Selecting actions is "espilon-greedy"
        # This means with probability epsilon we pick a random action, otherwise pick action with highest Q-value
        # At evaluation, set to 0.0 to prevent random exploration
        eps = self.epsilon if epsilon is None else epsilon

        if random.random() < eps:
            # Pick random action of 21
            return random.randint(0, 20)

        # Otherwise ask the network for Q-values
        q_values = self.online_net(obs[np.newaxis], training=False)  # (1, 21)
        # tf.argmax takes largest Q-value index over 21 actions
        return int(tf.argmax(q_values[0]).numpy())

    def store(self, obs, action, reward, next_obs, done):
        # push results to the reply buffer
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self):
        # Prevent training until buffer has reached threshold size
        if len(self.replay_buffer) < MIN_BUFFER_SIZE:
            return

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Compute targets using Double DQN
        # A Double DQN avoids the problem of overestimating Q-values in a noisy environment
        # Use online_net to select best action, then target_net to evaluate it

        # online net picks the best next action
        next_q_online = self.online_net(next_obs, training=False)
        best_next_actions = tf.argmax(next_q_online, axis=1)

        # target net evaluates that action
        next_q_target = self.target_net(next_obs, training=False)
        target_indices = tf.stack([tf.range(BATCH_SIZE), tf.cast(best_next_actions, tf.int32)], axis=1)
        max_next_q = tf.gather_nd(next_q_target, target_indices)

        # Training targets
        td_targets = rewards + GAMMA * max_next_q * (1.0 - dones)

        # Compute loss and backpropigation using online network

        # GradientTape records every action so it can compute gradients later, records forward pass basically
        # It records inputs -> model -> outputs -> loss
        # At the end you call tape.gradient() to get weight updates
        with tf.GradientTape() as tape:
            q_values = self.online_net(obs, training=True)  # (batch, 21)

            # Extract the one value actually taken per sample
            # tf.stack() builds a list of (sample, action) pairs for the Q values chosen
            # tf.gather_nd uses these indexes to select actual Q-value
            indices = tf.stack([tf.range(BATCH_SIZE), actions], axis=1)
            q_taken = tf.gather_nd(q_values, indices)

            # td_targets is what we want, q_taken is what we predicted
            # Take Huber loss of these
            loss = self.huber_loss(td_targets, q_taken)

        # Compute gradient
        # Using Adam optimizer, apply updates
        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        # After every update increment 
        self.steps_done += 1

    def decay_epsilon(self):
        # Called at end of each episode to linearly decrease epsilon
        # EPSILON_END makes sure we do not go below minimum value of 0.05
        self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)

    def sync_target(self):
        # Replace target_nets weights with updated online_net weights
        self.target_net.set_weights(self.online_net.get_weights())

    def save(self, path='dqn_weights.weights.h5'):
        # Save network to file so I dont have to rerun training over and over
        # Also save epsilon and steps_done in case I want to resume training
        self.online_net.save_weights(path)
        np.save(path + '_state.npy', np.array([self.epsilon, self.steps_done]))

    # Needs .weights.h5 extension
    def load(self, path='dqn_weights.weights.h5'):
        # Load weights and synch target network so both start from same weights
        self.online_net.load_weights(path)
        self.sync_target()
        state = np.load(path + '_state.npy')
        self.epsilon = float(state[0])
        self.steps_done = int(state[1])
