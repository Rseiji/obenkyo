"""Frozen Lake Agent implementation for studying purposes.

Adaption from doc's blackjack game implementation:
https://gymnasium.farama.org/introduction/train_agent/
"""

from collections import defaultdict

import gymnasium as gym
import numpy as np
from tqdm import tqdm


class FrozenLakeAgent:
    """Frozen lake agent."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        **kwargs,
    ) -> None:
        """
        Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate, and an epsilon.

        Parameters
        ----------
        env : gym.Env
            The training environment.
        learning_rate : float
            The learning rate.
        initial_epsilon : float
            The initial epsilon value.
        epsilon_decay : float
            The decay for epsilon.
        final_epsilon : float
            The final epsilon value.
        discount_factor : float, optional
            The discount factor for computing the Q-value, by default 0.95.
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        # self.q_values[15] = np.ones(env.action_space.n)

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ) -> None:
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self, episode_number):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def iterate_with_agent(
    agent: FrozenLakeAgent,
    environment: gym.Env,
    n_episodes: int
) -> FrozenLakeAgent:
    for episode in tqdm(range(n_episodes)):
        obs, info = environment.reset()
        done = False

        episode_over = False
        while not episode_over:
            action = agent.get_action(obs)
            next_observation, reward, terminated, truncated, info = environment.step(action)

            agent.update(
                obs,
                action,
                reward,
                terminated,
                next_observation,
            )

            obs = next_observation
            episode_over = terminated or truncated

        agent.decay_epsilon(episode_number=episode)

    environment.close()
    return agent


def render_trained_agent(
    agent: FrozenLakeAgent,
    map_name: str,
    max_episode_steps: int,
    n_episodes: int = 5
) -> None:
    render_env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=map_name,
        render_mode='human',
        is_slippery=False,
        max_episode_steps=max_episode_steps,
    )

    for _ in range(n_episodes):
        obs, info = render_env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated

    render_env.close()


def plot_q_matrix(q_matrix: np.array) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(q_matrix, annot=True, cmap="viridis", cbar=True)
    plt.title("Q-Value Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")


def plot_avg_state_qvalues(q_matrix: np.array) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_q_values, annot=True, cmap="viridis", cbar=True)
    plt.title("Average Q-Value Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")


if __name__ == "__main__":
    import frozen_lake as fl
    import gymnasium as gym
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.use('TkAgg')

    # configs = {
    #     "map_name": "4x4",
    #     "learning_rate": 0.7,
    #     "n_episodes": 10_000,
    #     "initial_epsilon": 1.0,
    #     "final_epsilon": 0.03,
    #     "epsilon_decay": 0.0005,
    #     "discount_factor": 0.95,
    # }

    configs = {
        "map_name": "8x8",
        "learning_rate": 0.95,
        "n_episodes": 100_000,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.15,
        "epsilon_decay": 0.00005,
        "discount_factor": 0.95,
        "max_episode_steps": 500,
    }

    map_name = configs["map_name"]
    n_episodes = configs["n_episodes"]
    max_episode_steps = configs["max_episode_steps"]

    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=map_name,
        # Way too bad to train as True
        # 4x4: becomes addicted to go to the wall!!
        # 8x8: goes nowhere!!!
        # is_slippery=True,
        is_slippery=False,
        # render_mode='human'
    )

    obs, info = env.reset()

    frozen = FrozenLakeAgent(env, **configs)
    frozen = iterate_with_agent(frozen, env, n_episodes)


    q_matrix = np.zeros((env.observation_space.n, env.action_space.n))
    env_dimensions = int(np.sqrt(env.observation_space.n))
    for state, actions in frozen.q_values.items():
        q_matrix[state] = actions
    # Calculate athe average Q-value for each stte
    avg_q_values = np.mean(q_matrix, axis=1).reshape((env_dimensions, env_dimensions))

    plot_q_matrix(q_matrix)
    plot_avg_state_qvalues(q_matrix)
    plt.show()

    render_trained_agent(frozen, map_name, max_episode_steps, n_episodes=5)
