# In a normal 4 bandit (one machine) problem, we see that the action on which we get a reward,
# we go on repeating that. i.e the same action is repeated again and again.

# In a multiple machine problem with rewards on different actions, we cannot just take the previous action
# that provided us with the reward as if it provides reward for one machine dosen't necessarily mean that same action provides
# reward for other machines as well.

from armedBandit import *
from NNagent import *


learning_rate = 0.01

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