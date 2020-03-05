import numpy as np
import pandas
import re

from utils import sigmoid, eating_hours, commissions, N_SPECILIATIES

class Restaurant():

    def __init__(self, id, name, lat, lon, specialities, seed=None):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon

        self.specialities = specialities # Array of the restaurant specialities 
        self.specialities_vec = np.zeros(N_SPECILIATIES)
        for spec in self.specialities:
            self.specialities_vec[spec] = 1

        self.rd = np.random.RandomState(seed=seed)
        self.current_day = 0
        self.grades = [] # Array of grades given by the users
        self.occupation = []
        self.day = [] # Day id for each given grade
    
    def generate_random_params(self, mean_n_tables):
        self.n_tables = np.maximum(int(self.rd.normal(mean_n_tables, mean_n_tables/2)), 2)
        self.tables = np.array([self.rd.choice([2,3,4,5,6]) for i in range(self.n_tables)])
        self.locked_tables = np.zeros(self.n_tables, dtype=np.bool)
        self.mean_price = (self.rd.beta(2, 5)+0.05) * 100
        self.is_lf_customer = self.rd.random() > 0.5 # Is the restaurant a La Fourchette customer ?
        self.exigency =  np.maximum(self.rd.normal(9, 4), 0.01) # Exigency of the restaurant about the commission taken by La Fourchette
        self.notoriety = self.rd.uniform(0.05, 0.95)
        self.horaires = [12, 19, 20, 22]
        self.deserved_grade = (self.rd.beta(6, 2)) * 5 # The grade the restaurant deserves
        
    # Load restaurant from a database (csv)
    @staticmethod
    def load_from_csv(path, set_random_params = False, mean_n_tables = 4, seed = None):
        df = pandas.read_csv(path)
        restaurants = []
        all_specs = []
        ind = 0

        for i in range(len(df)):
            specs = re.split(",|;", df["speciality"][i])
            all_specs.append(specs)
        specialities = np.unique(np.concatenate(all_specs))
        sp_dict = {sp: i for i, sp in enumerate(specialities) if sp != "na"}
        all_specs = [(ind, [sp_dict[sp] for sp in sps]) for ind, sps in enumerate(all_specs) if sps[0] != "na"]

        for i, specs in all_specs:

            restaurant = Restaurant(ind, df["name"][i], df["lat"][i], df["long"][i], specs, seed)
            if set_random_params:
                restaurant.generate_random_params(mean_n_tables)
            restaurants.append(restaurant)
            ind += 1
            
        return restaurants, sp_dict
    
    # Return restaurant specialities
    def get_specialities(self):
        return self.specialities

    # Update the restaurant notoriety 
    def update_notoriety(self):
        n_days = 2
        n_days = min(n_days, len(self.grades))

        total_non_satisfied = np.sum(self.grades == 0) # Total number of unsatisfied users
        total_satisfied = np.sum(self.grades == 5) # Total number of satisfied users
        last_grades = np.array(self.grades)[np.array(self.day) > self.current_day - n_days] # Last 2 days grades
        last_non_satisfied = np.sum(last_grades == 0) # Last 2 days unsatisfied users
        last_satisfied = np.sum(last_grades == 5) # Last 2 days satisfied users

        if len(last_grades) == 0:
            self.notoriety = max(0, self.notoriety - 0.05) # If no customer, decrease notoriety
        else:
            alpha = (last_satisfied - last_non_satisfied) / len(self.grades) - (total_satisfied - total_non_satisfied) / n_days
            self.notoriety = np.clip(self.notoriety + alpha, 0, 1)
        return alpha
    
    # Save current occupation of the restaurant
    def save_occupation(self):
        n_places = np.sum(self.tables)
        pourcentage = np.sum([self.tables[i] for i, taken in enumerate(self.locked_tables) if taken]) / n_places
        self.occupation.append(pourcentage)

    # Save a grade given by a user
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
    
    # Return true if the restaurant accepte the commission 
    def accept_commission(self, commission):
        return self.rd.rand() < self.staying_prob(commission)
    
    # Return the max commission the restaurant will accept
    def max_commission_accepted(self):
        accepted = [self.accept_commission(c) for c in commissions]
        return commissions[np.argmax(accepted)]
        
    # Return true if the restaurant has a table with enough places
    def can_welcome(self, n_people):
        t = self.tables[~self.locked_tables]
        return np.max(t) - n_people > 0 

    # Return a table and book it if book is equal to True
    def get_table(self, n_people, book=True):
        delta_places = [n_chaires - n_people for i, n_chaires in enumerate(self.tables) if not self.locked_tables[i]]
        delta_places = np.argsort([n_c for n_c in delta_places if n_c >= 0])
    
        if len(delta_places) > 0:
            table = delta_places[0]
            if book: 
                self.lock_table(table)
            return table
        else:
            return None

    # Lock a table (nobody else can book it)
    def lock_table(self, id_table):
        self.locked_tables[id_table] = 1
    
    # Free all the tables of the restaurant
    def free_tables(self):
        self.locked_tables = np.zeros(self.n_tables, dtype=np.bool)

    def get_feature_vec(self):
        last_grades = np.zeros(10) + 3
        n = min(10, len(self.grades))
        if n > 0:
            last_grades[-n:] = self.grades[-n:]

        last_occup = np.zeros(10) + 0.5
        n = min(10, len(self.occupation))
        if n > 0:
            last_occup[-n:] = self.occupation[-n:]
            
        price = self.mean_price

        return np.concatenate((last_grades, [price], last_occup, self.specialities_vec))
