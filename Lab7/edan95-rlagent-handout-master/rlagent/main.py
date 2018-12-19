"""main.py: The entrypoint of the program with the training loop."""
__author__ = "Erik Gärtner"

from time import time

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .agent import Agent


def env_solved(history):
    """
    Checks if the agent has solved the CartPole-v1 environment.
    """
    return history[-1]['mean_reward'] >= 450


def main(nbr_eps, tf_session, render_every=100):
    """Main training loop of the RL agent."""

    # Create the environment
    env = gym.make('CartPole-v1')
    env.seed(1)

    # Create the agent
    agent = Agent(tf_session,
                  state_size=env.observation_space.shape,
                  action_size=env.action_space.n,
                  learning_rate=1e-3,
                  gamma=0.99)
    history = []

    for ep in range(1, nbr_eps + 1):
        # Start an episode of training
        state0 = env.reset()
        done = False
        step = 0
        total_reward = 0
        tt = time()

        # Take actions until the end of the episode
        while not done:

            # Render every nth episode
            if ep % render_every == 0:
                env.render()

            # Ask agent for action
            action = agent.take_action(state0)

            # Apply action to the environment. Get reward and the new state.
            state1, reward, done, _ = env.step(action)

            # Record the <s, a, r, s'> tuple for later when training.
            agent.record_action(state0, action, reward, state1, done)

            total_reward += reward
            step += 1
            state0 = state1

        # Train the network after each episode
        agent.train_agent()

        # Save information about the episode
        mean_reward = np.mean(np.array([h['reward'] for h in history[-100:]])) \
            if len(history) > 0 else 0
        history.append({
            'ep': ep, 'reward': total_reward, 'steps': step, 'time': time() - tt,
            'mean_reward': mean_reward
        })

        # Print information about the episode
        print(ep, total_reward, step, mean_reward)

        if env_solved(history):
            print('Agent solved the environment in {} episodes!'.format(ep))
            plt.plot([h['mean_reward'] for h in history])
            plt.ylabel('Mean reward')
            plt.xlabel('Episode')
            plt.show()
            break


if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Session() as tf_session:
        main(3000, tf_session)
