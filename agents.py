import numpy as np

class BaseAgent():

    def __init__(self, seed):
        self.rd = np.random.RandomState(seed=seed)

    def init(self, seed):
        self.rd = np.random.RandomState(seed=seed)

    def get_action(self, user, restaurants):
        pass
    
    def update(self):
        pass

class RandomAgent(BaseAgent):

    def __init__(self, seed=None):
        super(RandomAgent, self).__init__(seed)

    def get_action(self, user, restaurants):
        n_restaurants = len(restaurants)
        if n_restaurants <= 0:
            return None
        return restaurants[self.rd.randint(0, n_restaurants)].id

class EpsGreedyAgent(BaseAgent):
    def __init__(self, epsilon, seed=None):
        super(EpsGreedyAgent).__init__(seed=seed)
        self.epsilon = epsilon

    def get_action(self, user, restaurants):
        n_restaurants = len(restaurants)
        p = self.rd.rand()
        if p < self.epsilon:
            ...
        else:
            return restaurants[self.rd.randint(0, n_restaurants)].id

class UCBAgent(BaseAgent):
    def __init__(self, seed=None):
        super(UCBAgent).__init__(seed)