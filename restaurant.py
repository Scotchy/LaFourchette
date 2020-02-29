import numpy as np
import pandas

from utils import sigmoid

class Restaurant():

    def __init__(self, name, lat, lon, speciality, seed=None):
        self.name = name 
        self.lat = lat
        self.lon = lon
        self.speciality = speciality.split(";")
        self.rd = np.random.RandomState(seed=seed)
        self.current_day = 0
        self.grades = []
        self.day = []
    
    def generate_random_params(self, mean_n_tables):
        self.n_tables = np.maximum(int(self.rd.normal(mean_n_tables, mean_n_tables/2)), 2)
        self.tables = [self.rd.choice([2,3,4,5,6]) for i in range(self.n_tables)]
        self.mean_price = (self.rd.beta(2, 5)+0.05) * 100
        self.exigency =  np.maximum(self.rd.normal(9, 4), 0.01)
        self.notoriety = self.rd.uniform(0.05, 0.95)
        self.horaires = [12, 19, 20, 22]
        self.is_lf_customer = self.rd.random() > 0.5

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
    
    def update_notoriety(self):
        total_non_satisfied = np.sum(self.grades == 0)
        total_satisfied = np.sum(self.grades == 5)
        last_grades = self.grades[self.day > self.current_day - 5]
        last_non_satisfied = np.sum(last_grades == 0)
        last_satisfied = np.sum(last_grades == 5)

        if len(last_grades) == 0:
            self.notoriety = max(0, self.notoriety - 0.05)
        else:
            alpha = last_satisfied / last_non_satisfied - total_satisfied / total_non_satisfied
            self.notoriety = np.clip(self.notoriety + alpha, 0, 1)
        return alpha
    
    def give_grade(self, grade):
        self.grades.append(grade)
        self.day.append(self.current_day)
    
    def staying_prob(self, commission):
        # commission € {0.05, 0.1, 0.15, 0.20, 0.25, 0.30}
        C = 1 / self.exigency
        a = (-3 + np.sqrt(9+120*C)) / 20
        b = 1 - C / a
        v = -self.notoriety + C / (commission + a) + b
        return sigmoid(20*v)

    def entering_prob(self, commission):
        # commission € {0.05, 0.1, 0.15, 0.20, 0.25, 0.30}
        C = 1 / self.exigency / 2
        a = (-3 + np.sqrt(9+120*C)) / 20
        b = 1 - C / a
        v = -self.notoriety + C / (commission + a) + b
        return sigmoid(5*v)

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