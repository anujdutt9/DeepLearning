# N-Armed Bandit Initializations and Functions

import numpy as np


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

# -------------------- EOC ---------------------