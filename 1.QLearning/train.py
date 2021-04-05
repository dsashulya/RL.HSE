from gym import make
import numpy as np
from typing import NoReturn
import matplotlib.pyplot as plt


GAMMA = 0.98
GRID_SIZE_X = 80
GRID_SIZE_Y = 60



# simple discretisation
def transform_state(state: np.ndarray) -> tuple:
    high = np.array([0.6 , 0.07])
    low = np.array([-1.2 , -0.07])
    step = np.divide(high - low,
                     np.array([GRID_SIZE_X, GRID_SIZE_Y]))
    return tuple(np.divide(state - low, step).astype(int))


class QLearning:
    def __init__(self, state_dim: list, action_dim: list,
                 learning_rate: float, discounting_rate: float):
        # creates a (GRID_SIZE_X x GRID_SIZE_Y x actions) random Q-table
        self.qlearning_estimate = np.random.uniform(-2, -1, (state_dim + action_dim))
        # self.qlearning_estimate = np.zeros((state_dim + action_dim))
        self.alpha = learning_rate
        self.gamma = discounting_rate

    def update(self, transition: tuple, achieved_goal=False) -> NoReturn:
        # Q-learning greedy update
        state, action, next_state, reward, done = transition
        state, next_state = transform_state(state), transform_state(next_state)
        if not done:
            self.qlearning_estimate[state][action] = (1 - self.alpha) * self.qlearning_estimate[state][action] +\
                    self.alpha * (reward + self.gamma * np.max(self.qlearning_estimate[next_state]))
        if achieved_goal:
            self.qlearning_estimate[state][action] = 0

    def act(self, state: np.ndarray) -> int:
        state = transform_state(state)
        return np.argmax(self.qlearning_estimate[state])

    def save(self, path="agent.npy"):
        np.save(path, self.qlearning_estimate)


def evaluate_policy(agent, episodes=5):
    env = make("MountainCar-v0")
    returns = []
    for episode in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            # env.render()
            state, reward,  done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("MountainCar-v0")

    ql = QLearning(state_dim=[GRID_SIZE_X, GRID_SIZE_Y], action_dim=[3],
                   learning_rate=0.1, discounting_rate=GAMMA)

    episodes = 50_000
    eps = 0.5
    for episode in range(episodes):
        env.seed(10)
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            if np.random.rand() <= eps:
                action = env.action_space.sample()
            else:
                action = ql.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            ql.update((state, action, next_state, reward, done))

            if next_state[0] >= env.goal_position:
                ql.update((state, action, next_state, reward, done), achieved_goal=True)

            state = next_state

        if episode % 250 == 0:
            rewards = evaluate_policy(ql, 5)
            print(f"Step: {episode+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            ql.save()


