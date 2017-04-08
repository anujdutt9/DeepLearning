# Agent Neural Network

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected


learning_rate = 0.01

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

# -------------------- EOC -----------------------