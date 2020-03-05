import numpy as np
from utils import sigmoid, eating_hours, N_SPECILIATIES, min_lf_prob

class User:
    
    def __init__(self, seed=None):
        self.rd = np.random.RandomState(seed)
        self.current_day = 0
        
        self.has_eaten = [(0, 0)]
        
        # Historical data
        self.restaurants = []
        self.given_grades = []
        self.money_spent = []
    
    def generate_random_params(self, restaurants, specialities, spacial_var=0.1):
        self.age = np.clip((self.rd.beta(2, 5)+0.05) * 100, 18, 100)
        restau = restaurants[self.rd.randint(len(restaurants))]
        self.lat = restau.lat + self.rd.normal(0, spacial_var)
        self.lon = restau.lon + self.rd.normal(0, spacial_var)
        self.n_people = int(self.rd.uniform(1, 5))

        self.price_appeatance = np.clip(sigmoid((self.age-15) / 20) * 100 + self.rd.normal(0, 10), 5, 100)
        self.use_lf_prob = [self.rd.uniform(min_lf_prob, 0.98)]
        self.promo_sensitivity = np.clip(self.rd.normal(0.3, 0.05), 0.2, 0.4)
        self.has_car = self.rd.rand() < 0.8 if self.age > 28 else self.rd.rand() < 0.1
        self.max_distance = np.clip(self.rd.normal(2000, 500) if self.has_car else self.rd.normal(5000, 2000), 0, 10000)
        self.preferences = self.rd.beta(0.3,0.3, len(specialities))
        self.price_sensitivity = self.rd.uniform(0.02, 0.1)
        self.accept_threshold = self.rd.uniform(0.005, 0.002)
        self.next_eating_day = self.rd.randint(0, 7)
        self.next_eating_hour = self.rd.choice(eating_hours)
        self.restaurant_frequency_per_day = self.rd.uniform(0.4, 0.8)
    
    """def compute_lf_prob(self, update=False):
        # Update the probability that the user will use La Fourchette next time
        n_last_exp = 3
        n_last = min(n_last_exp, len(self.given_grades)-1)
        new_prob = np.mean(self.given_grades[-n_last:]) / 5
        new_prob -= 0.2 * np.sum(self.given_grades[-n_last:] == 5) # If a customer really likes a restaurant, he will stop using La Fourchette
        new_prob -= 0.05 * np.sum(self.given_grades[-n_last:] == 5) # If a customer did not find a restaurant, he will stop using La Fourchette
        new_prob = np.clip(new_prob, 0, 1)
        if update:
            self.use_lf_prob.append(new_prob)
            self.n_people = int(self.rd.uniform(1, 5)) # Update the number of people next time
        return new_prob"""
    
    def compute_lf_prob(self, grade, update=False):
        if grade is not None:
            n_last_exp = 3
            n_last = min(n_last_exp, len(self.given_grades)-1)
            grades = self.given_grades[-n_last:] + [grade]

            new_prob = np.mean(grades) / 5
            new_prob -= 0.2 * np.sum(grades == 5) # If a customer really likes a restaurant, he will stop using La Fourchette
            new_prob -= 0.05 * np.sum(grades == 5) # If a customer did not find a restaurant, he will stop using La Fourchette
            new_prob = np.clip(new_prob, min_lf_prob, 1)
        else:
            new_prob = np.clip(self.use_lf_prob[-1] * 0.95, min_lf_prob, 1)
        
        if update:
            self.use_lf_prob.append(new_prob)
            self.n_people = int(self.rd.uniform(1, 4)) # Update the number of people next time
        return new_prob

    def update(self, grade):
        #self.lat += self.rd.normal(0, 0.005)
        #self.lon += self.rd.normal(0, 0.005)
        if grade != -1:
            self.compute_lf_prob(grade, update=True)
        self.plan_next_restaurant(update=True)

    def plan_next_restaurant(self, lf_prob=None, update=False):
        if lf_prob == None: 
            delta_days = self.rd.geometric(self.restaurant_frequency_per_day * self.use_lf_prob[-1])
        else:
            delta_days = self.rd.geometric(self.restaurant_frequency_per_day * lf_prob)
        if update:
            self.next_eating_day += delta_days
            self.next_eating_hour = self.rd.choice(eating_hours)
        return delta_days
    
    def get_lf_prob(self, promo=0): 
        p = sigmoid(10 * (promo - self.promo_sensitivity))
        return min(self.use_lf_prob[-1] + (p if p > 0.1 else 0), 1)

    def chose_restaurant(self, restaurants):
        evaluations = np.zeros(len(restaurants))

        for i, restaurant in enumerate(restaurants):
            # Calculate the mean of the last 10 grades of the restaurant
            last_grades = restaurant.grades
            if len(last_grades) == 0:
                mean_last_grade = 3
            else:
                mean_last_grade = np.mean(last_grades[-10:])
            
            # Calculate the difference between the price and price appeatance
            delta_price = self.price_appeatance - restaurant.mean_price
            if delta_price < 0:
                delta_price = 2 * np.abs(delta_price)
            
            # Calculate the distance cost
            distance = np.sqrt((self.lat - restaurant.lat) ** 2 + (self.lon - restaurant.lon) ** 2) * 50000
            distance = 1 if distance < self.max_distance else np.exp(10 * (self.max_distance - distance))

            # Evaluate the desire of going to this type of restaurant
            specialities = restaurant.get_specialities()
            desire = np.clip(np.max(self.preferences[specialities] + self.rd.normal(0, 0.1)), 0, 1)

            # Give the final evaluation
            evaluation = mean_last_grade / 5 * np.exp(- self.price_sensitivity * delta_price) * distance * desire
            evaluations[i] = evaluation

        best_ind = np.argmax(evaluations)
        if evaluations[best_ind] > self.accept_threshold:
            # If a restaurant is satisfying enough, we book the table
            table = restaurants[best_ind].get_table(self.n_people)
            return restaurants[best_ind].id, table
        else:
            return None, None
    
    def eat_at_restaurant(self, restaurant, id_table):
        if restaurant is None:
            self.restaurants.append(None)
            self.given_grades.append(-1)
            return -1

        money_spent = np.sum([np.clip(self.rd.normal((restaurant.mean_price + self.price_appeatance) / 2, 5), 4, 500) for i in range(self.n_people)])
        grade = np.clip(restaurant.deserved_grade + self.rd.normal(0, 1.2 * sigmoid(money_spent - self.price_appeatance)), 0, 5)
        return grade, money_spent
    
    def simulate_eating_at_restaurants(self, restaurants, restaurant_id, id_table):
        id_tables = [r.get_table(self.n_people, book=False) for r in restaurants]
        exp = np.array([self.eat_at_restaurant(r, id_t) for r, id_t in zip(restaurants, id_tables)])
        num_chaires = np.array([r.tables[id_t] if id_t is not None else np.inf for r, id_t in zip(restaurants, id_tables)])
        lf_probs = [self.compute_lf_prob(grade) for grade, money in exp]
        delta_days = np.array([self.plan_next_restaurant(lf_prob=prob) for prob in lf_probs])
        commissions = [r.max_commission_accepted() for r in restaurants]

        rewards = exp[:, 1] / delta_days * self.n_people / num_chaires * commissions
        max_reward = np.max(rewards)
        grades = exp[:, 0]
        if restaurant_id is not None and restaurant_id >= 0:
            restaurant_id = [i for i, r in enumerate(restaurants) if r.id == restaurant_id][0]
            self.money_spent.append(exp[restaurant_id][1])
            return rewards[restaurant_id], max_reward, grades[restaurant_id]
        else:
            return 0, max_reward, None


    def give_grade(self, restaurant, grade):
        self.given_grades.append(grade)
        self.restaurants.append(restaurant.id)
        restaurant.give_grade(grade)

    def wanna_eat(self, hour):
        
        if hour >= 11 and hour <= 14:
            # User did not eat at lunch and use La Fourchette
            return not self.has_eaten[-1][0] and self.rd.rand() < self.use_lf_prob[-1]
        elif hour >= 18 and hour <= 22:
            # User did not eat at diner and use La Fourchette
            return not self.has_eaten[-1][1] and self.rd.rand() < self.use_lf_prob[-1]
        else:
            return False
    
    def get_feature_vec(self):
        last_grades = np.zeros(10) + 3
        n = min(10, len(self.given_grades))
        if n > 0:
            last_grades[-n:] = self.given_grades[-n:]

        last_spent = np.zeros(10) + 0.5
        n = min(10, len(self.money_spent))
        if n > 0:
            last_spent[-n:] = self.money_spent[-n:]
        
        last_spec = np.zeros(N_SPECILIATIES)
        n = min(10, len(self.restaurants))
        if n > 0:
            specs = np.concatenate([r.get_specialities() for r in self.restaurants[-n:]])
            for sp in specs:
                last_spec[sp] += 1
            last_spec /= n

        return np.concatenate((last_grades, last_spent, last_spec))