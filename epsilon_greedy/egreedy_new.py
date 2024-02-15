import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Bandit:

    def __init__(self, arms=3, epsilon=0.1, true_reward=0, loop=1000, initial = 0):
        self.k = arms
        self.epsilon = epsilon
        self.true_rewards = np.random.randn(self.k) + true_reward
        self.estimates = np.full(self.k, 0)
        #count the number of times an action has been selected.
        self.action_count = np.zeros(self.k)
        #arranging the k arms as an numpy array for easier slection later.
        self.indices = np.arange(self.k)
        self.opportunities = loop


    def selectAction_epsilon(self):
        #selecting action based on the epsilon value.
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.indices)
        else:
            return np.argmax(self.estimates)

    def updateEstimates(self):
        action = self.selectAction_epsilon()
        reward = np.random.randn() + self.true_rewards[action]
        self.action_count[action] += 1
        self.estimates[action] += (1/self.action_count[action])*(reward - self.estimates[action])
        return reward

    def maximise_epsilon(self):
        reward = 0
        for _ in range(self.opportunities):
            reward += self.updateEstimates()
        return reward/self.opportunities

def main():
    """b1, b2, and b3 compares the results for three different epsilon value."""
    b1 = Bandit(10, 0, 5, 1000, 0)
    b2 = Bandit(10, 0.01, 5, 1000, 0)
    b3 = Bandit(10, 0.1, 5, 1000, 0)
    num_of_bandits = 3
    num_of_iterations = 2000
    reward = np.zeros((num_of_iterations, 3))
    for i in tqdm(range(num_of_iterations)):
        reward[i][0] = b1.maximise_epsilon()
        reward[i][1] = b2.maximise_epsilon()
        reward[i][2] = b3.maximise_epsilon()

    x = np.arange(num_of_iterations)
    plt.plot(x, reward[:, 0])
    plt.plot(x, reward[:, 1])
    plt.plot(x, reward[:, 2])
    plt.title("Comparing greedy and e-greedy action value methods.")
    plt.xlabel("Steps")
    plt.ylabel("Average Rewards")
    plt.legend(['e=0','e=0.01','e=0.1'], loc='lower right')
    plt.show()
    """b4, b5, and b6 compares the model for three different initial values with same and different epsilons."""
    b4 = Bandit(10, 0, 5, 1000, 5)
    b5 = Bandit(10, 0.1, 5, 1000, 0) 
    """e greddy without the optimised initial values."""
    b6 = Bandit(10, 0.1, 5, 1000, 5) 
    """Optimised e greedy algorithm."""
    num_of_iterations = 200000
    reward = np.zeros((num_of_iterations, 3))
    for i in tqdm(range(num_of_iterations)):
        reward[i][0] = b4.maximise_epsilon()
        reward[i][1] = b5.maximise_epsilon()
        reward[i][2] = b6.maximise_epsilon()

    x = np.arange(num_of_iterations)
    plt.plot(x, reward[:, 0])
    plt.plot(x, reward[:, 1])
    plt.plot(x, reward[:, 2])
    plt.title("Comparing greedy and e-greedy action value methods.")
    plt.xlabel("Steps")
    plt.ylabel("Average Rewards")
    plt.legend(['e=0, i=5','e=0.1, i=0','e=0.1, i=5'], loc='lower right')
    plt.show()
    



if __name__ == '__main__':
    main()

