import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GradientBandit:

    def __init__(self, arms=3, epsilon=0.1, true_reward=0, n_iterations=1000, initial = 0, alpha=0.1, baseline=0):
        self.k = arms
        self.epsilon = epsilon
        """actual rewards which the agent does not know."""
        self.true_rewards = np.random.randn(self.k) + true_reward 
        self.estimates = np.full(self.k, initial)
        """count the number of times an action has been selected."""
        self.action_count = np.zeros(self.k)
        """arranging the k arms as an numpy array for easier selection later."""
        self.indices = np.arange(self.k)
        """number of iterations or the opportunities that the agent gets."""
        self.opportunities = n_iterations
        self.alpha = alpha
        """H(a) at time t. Numerical preference for ach action."""
        self.preferences = np.zeros(self.k)
        """Probability of taking an Action At = a. Initial probability for every action is same."""
        self.pi_a = np.full(self.k, 1/self.k)
        self.choice = baseline
        """Initializing the average reward."""
        self.avg_reward = 0
        """ Calculate the R dash i.e. the average of all the rewards up through and includeing time t."""
    def selectAction_epsilon(self):
        """selecting action based on the epsiIlon value."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.indices)
        else:
            return np.argmax(self.estimates)

    def selectAction_gradient(self):
        return np.random.choice(self.indices, 1, p=self.pi_a)


    def updateAvgReward(self, reward):
        #FUNCTION NOT USED.
        self.avg_reward = self.avg_reward + self.alpha * (reward - self.avg_reward)


    def updateEstimatesNonStat(self, action, reward):
        #Update the estimates 
        self.estimates[action] = self.estimates[action] + self.alpha * (reward - self.estimates[action])


    
    def updatePreferences(self, action, reward):
        for i in range(self.k):
            if i == action:
                self.preferences[i] = self.preferences[i] + (self.alpha * (reward - self.estimates[i]) * (1 - self.pi_a[i]))
            else:
                self.preferences[i] = self.preferences[i] - (self.alpha * (reward - self.estimates[i]) * self.pi_a[i])

    def updatePreferencesNoBaseline(self, action, reward):
        for i in range(self.k):
            if i == action:
                self.preferences[i] = self.preferences[i] + (self.alpha * (reward - 0) * (1 - self.pi_a[i]))
            else:
                self.preferences[i] = self.preferences[i] - (self.alpha * (reward - 0) * self.pi_a[i])
        

    def updateProbabilities(self):
        """Calculating the sum of all the exp(Ht(b)) terms"""
        exp_arms =  np.exp(self.preferences)
        total_exp = np.sum(exp_arms)
        """updating the individual probabilities"""
        #for i in range(self.k):
            #pi_a = exp_arms/total_prob
        self.pi_a = exp_arms/total_exp

    def takeAction(self):
        action = self.selectAction_gradient()
        reward = np.random.randn() + self.true_rewards[action]
        self.action_count[action] += 1
        self.updateEstimatesNonStat(action, reward) #average reward in the equation (from RL by Sutton and Barto).
        self.updateProbabilities()  #Updating probabilities based on the new preferences.
        if self.choice == 0:
            self.updatePreferences(action, reward)  #Updating preferences.
        else:
            self.updatePreferencesNoBaseline(action, reward)
        #self.estimates[action] += (1/self.action_count[action])*(reward - self.estimates[action])
        return reward


def main():
    gradient1 = GradientBandit(arms=10, true_reward=10, n_iterations=1000, initial=0, alpha=0.1)
    gradient2 = GradientBandit(arms=10, true_reward=10, n_iterations=1000, initial=0, alpha=0.4)
    gradient3 = GradientBandit(arms=10, true_reward=10, n_iterations=1000, initial=2, alpha=0.1)
    gradient4 = GradientBandit(arms=10, true_reward=10, n_iterations=1000, initial=0, alpha=0.1, baseline=1)
    num_of_bandits = 3
    num_of_iterations = 200000
    reward = np.zeros((num_of_iterations, 4))
    for i in tqdm(range(num_of_iterations)):
        reward[i][0] = gradient1.takeAction()
        reward[i][1] = gradient2.takeAction()
        reward[i][2] = gradient3.takeAction()
        reward[i][3] = gradient4.takeAction()


    x = np.arange(num_of_iterations)
    plt.title("Gradient Bandit Algorithm using the soft-max distribution.")
    plt.subplot(4, 1, 1)
    plt.plot(x, reward[:, 0], label="Gradient 1 with alpha = 0.1")
    plt.subplot(4, 1, 2)
    plt.plot(x, reward[:, 1], label="Gradient 2 with alpha = 0.4", color="Green")
    plt.subplot(4, 1, 3)
    plt.plot(x, reward[:, 2], label="Gradient 3 with alpha = 0.1 and initial = 2", color="orange")
    plt.subplot(4, 1, 4)
    plt.plot(x, reward[:, 3], label="Without baseline; alpha = 0.1", color="purple", alpha=1)
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Steps")
    plt.ylabel("Average Rewards")
    plt.show()

    print(np.sum(reward, axis=0))



if __name__ == '__main__':
    main()

"""
This is an example of non-stationary problem 
as the probabilities are not constant. They are updated after each step.
"""
