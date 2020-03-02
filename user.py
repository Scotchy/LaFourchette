import numpy as np
from utils import sigmoid

class User:
    
    def __init__(self, seed=None):
        self.rd = np.random.RandomState(seed)
        self.current_day = 0
        self.restaurants = []
        self.given_grades = []
        self.has_eaten = []
    
    def generate_random_params(self, restaurants, specialities, spacial_var = 0.1):
        self.age = np.clip((self.rd.beta(2, 5)+0.05) * 100, 18, 100)
        restau = restaurants[self.rd.randint(len(restaurants))]
        self.lat = restau.lat + self.rd.normal(0, spacial_var)
        self.lon = restau.lon + self.rd.normal(0, spacial_var)
        self.n_people = int(self.rd.uniform(1, 5))

        self.price_appeatance = np.clip(sigmoid((self.age-15) / 20) * 100 + self.rd.normal(0, 10), 5, 100)
        self.use_lf_prob = [self.rd.uniform(0, 0.98)]
        self.promo_sensitivity = np.clip(self.rd.normal(0.3, 0.05), 0.2, 0.4)
        self.has_car = self.rd.rand() < 0.8 if self.age > 28 else self.rd.rand() < 0.1
        self.max_distance = np.clip(self.rd.normal(2000, 500) if self.has_car else self.rd.normal(5000, 2000), 0, 10000)
        self.preferences = self.rd.beta(0.3,0.3, len(specialities))
        self.price_sensitivity = self.rd.uniform(0.02, 0.1)
        self.accept_threshold = self.rd.uniform(0.2, 0.55)
    
    def update_use_lf_prob(self):
        # Update the probability that the user will use La Fourchette next time
        n_last_exp = 3
        n_last = min(n_last_exp, len(self.given_grades)-1)
        new_prob = np.mean(self.given_grades[-n_last:]) / 5
        new_prob -= 0.2 * np.sum(self.given_grades[-n_last:] == 5) # If a customer really likes a restaurant, he will stop using La Fourchette
        new_prob -= 0.05 * np.sum(self.given_grades[-n_last:] == 5) # If a customer did not find a restaurant, he will stop using La Fourchette
        new_prob = np.clip(new_prob, 0, 1)
        self.use_lf_prob.append(new_prob)
        self.n_people = int(self.rd.uniform(1, 5)) # Update the number of people next time
        return new_prob
    
    def get_lf_prob(self, promo=0): 
        p = sigmoid(10 * (promo - self.promo_sensitivity))
        return min(self.use_lf_prob[-1] + (p if p > 0.1 else 0), 1)

    def chose_restaurant(self, restaurants, sp_dict):
        evaluations = np.zeros(len(restaurants))

        for i, restaurant in enumerate(restaurants):
            # Calculate the mean of the last 10 grades of the restaurant
            last_grades = restaurant.grades[-10:]
            if len(last_grades) == 0:
                mean_last_grade = 3
            else:
                mean_last_grade = np.mean(last_grades)
            
            # Calculate the difference between the price and price appeatance
            delta_price = self.price_appeatance - restaurant.mean_price
            if delta_price < 0:
                delta_price = 2 * np.abs(delta_price)
            
            # Calculate the distance cost
            distance = np.sqrt((self.lat - restaurant.lat) ** 2 + (self.lon - restaurant.lon) ** 2) * 50000
            distance = 1 if distance < self.max_distance else np.exp(10 * (self.max_distance - distance))

            # Evaluate the desire of going to this type of restaurant
            specialities = restaurant.get_specialities(sp_dict)
            desire = np.clip(np.max(self.preferences[specialities] + self.rd.normal(0, 0.1)), 0, 1)

            # Give the final evaluation
            evaluation = mean_last_grade / 5 * np.exp(- self.price_sensitivity * delta_price) * distance * desire
            evaluations[i] = evaluation

        best_ind = np.argmax(evaluations)
        if evaluations[best_ind] > self.accept_threshold:
            # If a restaurant is satisfying enough, we book the table
            table = restaurants[best_ind].get_table(self.n_people)
            return evaluations[best_ind], best_ind, table, self.n_people
        else:
            return evaluations[best_ind], None, None, self.n_people
    
    def eat_at_restaurant(self, restaurant, id_table):
        if restaurant is None:
            self.restaurants.append(None)
            self.given_grades.append(-1)
            return -1

        money_spent = np.sum([self.rd.normal(restaurant.mean_price, 10) for i in range(self.n_people)])
        grade = np.clip(restaurant.deserved_grade + self.rd.normal(0, 1.2*sigmoid(money_spent - self.price_appeatance)), 0, 5)
        return grade, money_spent
    
    def give_grade(self, restaurant, grade):
        self.given_grades.append(grade)
        self.restaurants.append(restaurant.id)
        restaurant.give_grade(grade)
    
    def wanna_eat(self, hour):
        if hour > 11 and hour < 14:
            # User did not eat at lunch and use La Fourchette
            return not self.has_eaten[-1][0] and self.rd.rand() < self.use_lf_prob
        elif hour > 18 and hour < 22:
            # User did not eat at diner and use La Fourchette
            return not self.has_eaten[-1][1] and self.rd.rand() < self.use_lf_prob
        else:
            return False