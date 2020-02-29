import numpy as np
from utils import sigmoid

class User:
    
    def __init__(self, lat, lon, seed=None):
        self.rd = np.random.RandomState(seed)
        self.lat = lat
        self.lon = lon
        self.current_day = 0
        self.restaurants = []
        self.given_grades = []
    
    def generate_random_params(self, specialities):
        self.age = np.clip((self.rd.beta(2, 5)+0.05) * 100, 18, 100)

        self.price_appeatance = np.clip(np.exp(self.age) / np.exp(100) * 100 + self.rd.normal(0, 10), 0, 100)
        self.use_lf_prob = [self.rd.uniform(0, 0.98)]
        self.promo_sensitivity = np.clip(self.rd.normal(0.3, 0.05), 0.2, 0.4)
        self.has_car = self.rd.rand() < 0.8 if self.age > 28 else self.rd.rand() < 0.1
        self.max_distance = np.clip(self.rd.normal(2000, 500) if self.has_car else self.rd.normal(5000, 2000), 0, 10000)
        self.preferences = self.rd.beta(0.3,0.3, 292)
    
    def update_use_lf_prob(self):
        n_last_exp = 3
        n_last = min(n_last_exp, len(self.given_grades)-1)
        new_prob = np.mean(self.given_grades[-n_last:]) / 5 - 0.2 * np.sum(self.given_grades[-n_last:] == 5)
        new_prob = np.clip(new_prob, 0, 1)
        self.use_lf_prob.append(new_prob)
        return new_prob
    
    def get_lf_prob(self, promo=0): 
        p = sigmoid(10 * (promo - self.promo_sensitivity))
        return min(self.use_lf_prob[-1] + (p if p > 0.1 else 0), 1)

    def chose_restaurant(self, restaurants):
        evaluations = np.zeros(len(restaurants))

        for i, restaurant in enumerate(restaurants):
            mean_last_grade = np.mean(restaurant.given_grades[-10:])
            delta_price = np.abs(self.price_appeatance - restaurant.mean_price)
            distance = np.sqrt((self.lat - restaurant.lat) ** 2 + (self.lon - restaurant.lon) ** 2)
            distance = 1 if distance < self.max_distance else np.exp(self.max_distance - distance)
            pref = ...
            evaluation = mean_last_grade / 5 * np.exp(-delta_price) * distance 
            evaluations[i] = evaluation
        best_ind = np.argmax(evaluations)
        if evaluations[best_ind] > 0.5:
            return evaluations[best_ind]
    
    def give_grade(self, restaurant):
        grade = 0
        self.given_grades.append(grade)
        self.restaurants.append(restaurant.id)
        restaurant.give_grade(grade)