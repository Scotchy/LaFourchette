import numpy as np
import pandas

class Restaurant():

    def __init__(self, name, lat, lon, speciality, seed=None):
        self.name = name 
        self.lat = lat
        self.lon = lon
        self.speciality = speciality
        self.rd = np.random.RandomState(seed=seed)
    
    def generate_random_params(self, mean_n_tables):
        self.n_tables = np.maximum(int(self.rd.normal(mean_n_tables, mean_n_tables/2)), 2)
        self.tables = [np.random.choice([2,3,4,5,6]) for i in range(self.n_tables)]
        self.mean_price = (np.random.beta(2, 5)+0.05) * 100
        self.horaires = [12, 19, 20, 22]

    @staticmethod
    def load_from_csv(path, set_random_params = False, mean_n_tables = 4, seed = None):
        df = pandas.read_csv(path)
        restaurants = []
        for i in range(len(df)):
            restaurant = Restaurant(df["name"][i], df["lat"][i], df["long"][i], df["speciality"][i], seed)
            if set_random_params:
                restaurant.generate_random_params(mean_n_tables)
            restaurants.append(restaurant)
        return restaurants
        
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