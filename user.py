import numpy as np

class User:
    
    def __init__(self, seed):
        self.rd = np.random.RandomState(seed)
        self.price_appeatance = (self.rd.beta(2, 5)+0.05) * 100
        self.use_lf_prob = [self.rd.uniform(0, 0.98)]
        self.restaurants = []
        self.given_grades = []
    
    def update_use_lf_prob(self):
        n_last = min(3, len(self.given_grades))
        new_prob = np.mean(self.given_grades[-1:-n_last]) / 5 - 0.2 * np.sum(self.given_grades[-1:-n_last] == 5)
        new_prob = np.clip(new_prob, 0, 1)
        self.use_lf_prob.append(new_prob)
        return new_prob