import numpy as np
import pandas
import re

from utils import sigmoid

class Restaurant():

    def __init__(self, id, name, lat, lon, specialities, seed=None):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.specialities = specialities # Array of the restaurant specialities 
        self.rd = np.random.RandomState(seed=seed)
        self.current_day = 0
        self.grades = [] # Array of grades given by the users
        self.day = [] # Day id for each given grade
    
    def generate_random_params(self, mean_n_tables):
        self.n_tables = np.maximum(int(self.rd.normal(mean_n_tables, mean_n_tables/2)), 2)
        self.tables = [self.rd.choice([2,3,4,5,6]) for i in range(self.n_tables)]
        self.locked_tables = np.zeros(self.n_tables)
        self.mean_price = (self.rd.beta(2, 5)+0.05) * 100
        self.is_lf_customer = self.rd.random() > 0.5 # Is the restaurant a La Fourchette customer ?
        self.exigency =  np.maximum(self.rd.normal(9, 4), 0.01) # Exigency of the restaurant about the commission taken by La Fourchette
        self.notoriety = self.rd.uniform(0.05, 0.95)
        self.horaires = [12, 19, 20, 22]
        self.deserved_grade = (self.rd.beta(6, 2)) * 5 # The grade the restaurant deserves
        

    @staticmethod
    def load_from_csv(path, set_random_params = False, mean_n_tables = 4, seed = None):
        df = pandas.read_csv(path)
        restaurants = []
        all_specs = []
        for i in range(len(df)):
            specs = re.split(",|;", df["speciality"][i])
            if specs[0] == "na":
                continue
            all_specs.append(specs)
            restaurant = Restaurant(i, df["name"][i], df["lat"][i], df["long"][i], specs, seed)
            if set_random_params:
                restaurant.generate_random_params(mean_n_tables)
            restaurants.append(restaurant)

        specialities = np.unique(np.concatenate(all_specs))
        sp_dict = {sp: i for i, sp in enumerate(specialities) }
        return restaurants, sp_dict
    
    def get_specialities(self, sp_dict=None):
        if sp_dict:
            return [sp_dict[sp] for sp in self.specialities]
        return self.specialities

    # Update the restaurant notoriety 
    def update_notoriety(self):
        n_days = 5
        n_days = min(n_days, len(self.grades))

        total_non_satisfied = np.sum(self.grades == 0) # Total number of unsatisfied users
        total_satisfied = np.sum(self.grades == 5) # Total number of satisfied users
        last_grades = np.array(self.grades)[np.array(self.day) > self.current_day - n_days] # Last 5 days grades
        last_non_satisfied = np.sum(last_grades == 0) # Last 5 days unsatisfied users
        last_satisfied = np.sum(last_grades == 5) # Last 5 days satisfied users

        if len(last_grades) == 0:
            self.notoriety = max(0, self.notoriety - 0.05) # If no customer, decrease notoriety
        else:
            alpha = (last_satisfied - last_non_satisfied) / len(self.grades) - (total_satisfied - total_non_satisfied) / n_days
            self.notoriety = np.clip(self.notoriety + alpha, 0, 1)
        return alpha
    
    def give_grade(self, grade):
        self.grades.append(grade)
        self.day.append(self.current_day)
    
    def staying_prob(self, commission):
        # Compute the probability that a restaurant stays a La Fourchette user
        # commission € {0.05, 0.1, 0.15, 0.20, 0.25, 0.30}
        C = 1 / self.exigency
        a = (-3 + np.sqrt(9+120*C)) / 20
        b = 1 - C / a
        v = -self.notoriety + C / (commission + a) + b
        return sigmoid(20*v)

    def entering_prob(self, commission):
        # Compute the probability of a restaurant to become a La Fourchette user
        # commission € {0.05, 0.1, 0.15, 0.20, 0.25, 0.30}
        max_entering_prob = 0.7
        C = 1 / self.exigency / 2
        a = (-3 + np.sqrt(9+120*C)) / 20
        b = 1 - C / a
        v = -self.notoriety + C / (commission + a) + b
        return min(sigmoid(5*v), max_entering_prob)
    
    def can_welcome(self, n_people):
        return np.max(self.n_tables) - n_people > 0 

    def get_table(self, n_people):
        delta_places = [n_chaires - n_people for i, n_chaires in enumerate(self.tables) if not self.locked_tables[i]]
        delta_places = np.argsort([n_c for n_c in delta_places if n_c >= 0])
    
        if len(delta_places) > 0:
            table = delta_places[0]
            self.lock_table(table)
            return table
        else:
            return None

    def lock_table(self, id_table):
        self.locked_tables[id_table] = 1
    
    def free_tables(self):
        self.locked_tables = 0

