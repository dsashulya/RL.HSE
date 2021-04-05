from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 4028
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


class DQN:
    def __init__(self, state_dim, action_dim, hidden_size, gamma, maxlen=10000, batch_size=128, lr=5e-4):
        self.steps = 0
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.target = copy.deepcopy(self.model)
        self.gamma = gamma

        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.optimiser = Adam(self.model.parameters(), lr=lr)
        self.score = -np.inf

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        batch = random.sample(self.buffer, self.batch_size)
        return list(zip(*batch))

    def train_step(self, batch):
        # Use batch to update DQN's network.
        
        state, action, next_state, reward, done = batch
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32).view(-1)
        done = torch.tensor(done)

        Q = self.model(state)[np.arange(self.batch_size).reshape(-1, 1), action].view(-1)
        with torch.no_grad():
            next_Q = self.target(next_state).max(dim=-1)[0]
        target = reward + self.gamma * torch.logical_not(done) * next_Q
        loss = F.mse_loss(Q, target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def update_target_network(self, tau=0.001):
        # Update weights of a target Q-network here.
        # for source, target in zip(self.model.parameters(),
        #                           self.target.parameters()):
        #     target.data.copy_(source.data)
        self.target = copy.deepcopy(self.model)
        # for source, target in zip(self.model.parameters(),
        #                           self.target.parameters()):
        #     target.data.copy_((1. - tau) * target.data + tau * source.data)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
        with torch.no_grad():
            action = self.model(state)
        return np.argmax(action.numpy())

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self, score):
        if score > self.score:
            torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0],
              action_dim=env.action_space.n,
              gamma=GAMMA, hidden_size=64)
    eps = 0.6
    eps_decay = eps / 200000

    env.seed(0)
    state = env.reset()
    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        env.seed(0)
        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        if 1 <= i <= 300000:
            eps -= eps_decay

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save(np.mean(rewards))
