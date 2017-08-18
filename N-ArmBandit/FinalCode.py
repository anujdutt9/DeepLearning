# In a normal 4 bandit (one machine) problem, we see that the action on which we get a reward,
# we go on repeating that. i.e the same action is repeated again and again.

# In a multiple machine problem with rewards on different actions, we cannot just take the previous action
# that provided us with the reward as if it provides reward for one machine dosen't necessarily mean that same action provides
# reward for other machines as well.

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import fully_connected


learning_rate = 0.01

# N - Armed Bandit Class
# Initialize all values in here
class N_Armed_Bandit(object):
    def __init__(self):
        Machine1 = [0.2, 0, -0.2, -5]
        Machine2 = [0.2, 0, -0, -5]
        Machine3 = [0.6, -5, 1, -0.50]
        Machine4 = [-0.7, 1, 2.5, 3]
        self.bandits = np.array([Machine1, Machine2, Machine3, Machine4])
        # Number of Bandits = size of np array bandits
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        # Define the state
        self.current_state = 0

    # Function to Generate a random state each time so that
    # the previous action does not give the reward for other machines
    def getRandomState(self):
        self.current_state = np.random.randint(0, len(self.bandits))
        return self.current_state

    # Function to Simulate the pulling of a Bandit
    def pullBandit(self, action):
        bandit = self.bandits[self.current_state, action]
        result = np.random.randn(1)
        # Check if pulling the Bandit results in a Reward or not
        if result > bandit:
            # If it results in a reward, return "1"
            return 1
        else:
            # If it results in a loss, return "-1"
            return -1



# Agent Class
class NNAgent(object):
    def __init__(self, lr, st_size, act_size):
        # Current State Placeholder
        self.current_state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        # Create a One hot vector
        current_state_OneHot = tf.one_hot(self.current_state_in, st_size)

        # Output of the Fully Connected Layer
        output = fully_connected(current_state_OneHot, act_size,
                                 activation_fn=tf.nn.sigmoid,
                                 biases_initializer=None,
                                 weights_initializer=tf.ones_initializer())

        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # Placeholder to store rewards
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])

        # Policy Gradient Calculation
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)

        # Define and use Gradient Descent Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # Update the Neural Network Agen values
        self.update = optimizer.minimize(self.loss)



# Main Function
def main():
    tf.reset_default_graph()

    bandit = N_Armed_Bandit()

    # Load the Neural Network Agent
    agent = NNAgent(lr=learning_rate, st_size=bandit.num_bandits, act_size=bandit.num_actions)
    # Initialize Weights
    W = tf.trainable_variables()[0]
    total_episodes = 1000
    # Initialize the Total Reward
    total_reward = np.zeros([bandit.num_bandits, bandit.num_actions])
    # Set the chance of taking a random action
    e = 0.1

    init = tf.global_variables_initializer()

    # Launch the Tensorflow Graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        while i < total_episodes:
            # Get the Randomly Generated State
            state = bandit.getRandomState()
            # Choose either a random action or one from our network
            if np.random.rand(1) < e:
                action = np.random.randint(bandit.num_actions)
            else:
                action = sess.run(agent.chosen_action, feed_dict={agent.current_state_in: [state]})
                # print('\nState: {}, Action Taken: {}'.format(state, action))

            reward = bandit.pullBandit(action)

            # Update the network
            _, ww = sess.run([agent.update, W], feed_dict={agent.reward_holder: [reward],
                                                           agent.action_holder: [action],
                                                           agent.current_state_in: [state]})

            # Update the Total Reward
            total_reward[state, action] += reward

            if i % 50 == 0:
                print('Running reward for the ' + str(bandit.num_bandits) + ' bandits:\n ' + str(total_reward))
                print('\nMean reward for the ' + str(bandit.num_bandits) + ' bandits:\n ' + str(np.mean(total_reward, axis=1)))
                print('\n--------------------------------\n')
            i += 1

    print('Predicted Action to Win the Game on Each Machine...\n')

    for i in range(bandit.num_bandits):
        if np.argmax(ww[i]) == np.argmin(bandit.bandits[i]):
            print('Actual Good Action: ', np.argmin(bandit.bandits[i])+1, ' for Machine ' + str(i+1))
            print('Agent Predicted Good Action: ',np.argmax(ww[i]) + 1, ' for Machine ' + str(i+1))
            print('Agent was Right !!!\n')
        else:
            print('----------------------------------\n')
            print('Actual Good Action: ', np.argmin(bandit.bandits[i]) + 1, ' for Machine ' + str(i + 1))
            print('Agent Predicted Good Action: ', np.argmax(ww[i] ) + 1, ' for Machine ' + str(i + 1))
            print('Agent was wrong !!!\n')



# --------------------------- Testing -------------------------
if __name__ == '__main__':
    main()

# -------------------- EOC ----------------------