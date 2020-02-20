import numpy as np
import pandas

class DataGen():

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def load_data(self, path):
        df = pandas.read_csv(path)
        return df.values

    def save_data(self, path):
        pass

    def gen_users_uniform(self, n_users, seed=None):
        pass