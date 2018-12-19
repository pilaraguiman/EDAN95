"""agent.py: Contains the entire deep reinforcement learning agent."""

from collections import deque

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .expreplay import ExpReplay


class Agent():
    """
    The agent class where you should implement the vanilla policy gradient agent.
    """

    def __init__(self, tf_session, state_size=(4,), action_size=2,
                 learning_rate=1e-3, gamma=0.99, memory_size=5000):
        """
        The initialization function. Besides saving attributes we also need
        to create the policy network in Tensorflow that later will be used.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.tf_sess = tf_session
        self.gamma = gamma
        self.replay = ExpReplay(memory_size)

        with tf.variable_scope('agent'): # The policy / network is pi. The decisionmaker follows the policy below.


            # Create tf placeholders, i.e. inputs into the network graph.
            self.input = tf.placeholder(shape=[None,state_size[0]],dtype=tf.float32)
            self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
            self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)


            # Create the hidden layers
            hidden = slim.fully_connected(self.input, 128, biases_initializer = None, activation_fn=tf.nn.relu)
            self.output = slim.fully_connected(hidden, action_size, activation_fn = tf.nn.softmax, biases_initializer = None)
            self.chosen_action = tf.argmax(self.output, 1)


            # Create the loss. We need to multiply the reward with the
            # log-probability of the selected actions.
            t_vars = tf.trainable_variables()
            self.indices = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)
            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)


            # Create the optimizer to minimize the loss
            self.gradients = tf.gradients(self.loss,t_vars)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.apply_gradients(zip(self.gradients,t_vars))
        tf_session.run(tf.global_variables_initializer())

    def take_action(self, state):
        """
        Given the current state sample an action from the policy network.
        Return a the index of the action [0..N).
        """
        action_dist = self.tf_sess.run(self.output, feed_dict = {self.input:[state]}) # Run the network. Returns action distribution from network (Probability for action 0 and 1, sum is 1)
        action = np.random.choice(action_dist[0], p=action_dist[0]) # Choose an action with probability from action_dist (Returns the probability of taken action
        action = np.argmax(action_dist == action) # Get the argument of the chosen action (0 or 1)
        return action

    def record_action(self, state0, action, reward, state1, done):
        """
        Record an action taken by the action and the associated reward
        and next state. This will later be used for traning.
        """
        self.replay.add([state0, action, reward, state1]) # Add the data in expreplay


    def train_agent(self): # Update the network
        """
        Train the policy network using the collected experiences during the
        episode(s).
        """
        # Retrieve collected experiences from memory
        experiences = np.array( self.replay.get_all()) # Next line only works on np array
        rewards = experiences[:, 2] # Get the 2nd column from experiences (The rewards)

        # Discount and normalize rewards
        discounted_rewards = np.zeros_like(rewards) # np.zeros does not work because reward is float
        R_t = 0 # Sum
        for timestamp in reversed(range(rewards.size)): # For reward for each timestamp in episode
            R_t = R_t * self.gamma + rewards[timestamp] # Gamma = discount_factor, value between 0 and 1 (.99) # Converges to 1/(1-gamma) when -> infinity (100 in this case)
            discounted_rewards[timestamp] = R_t

        discounted_rewards = discounted_rewards - np.mean(discounted_rewards[:]) # 0 mean
        discounted_rewards = discounted_rewards / np.std(discounted_rewards[:]) # 1 unit variance
        experiences[:, 2] = discounted_rewards

        # Shuffle for better learning
        #    np.random.shuffle(experiences[:, 2]) # This makes it not work at all :P

        # Feed the experiences through the network with rewards to compute and
        # minimize the loss.
        feed_dict = {self.reward_holder:experiences[:, 2], self.action_holder:experiences[:, 1], self.input:np.vstack(experiences[:, 0])}
        for i in range(3): # Train more than one time !! Worked well
            self.tf_sess.run(self.train_op, feed_dict = feed_dict)
        self.replay.clear()
