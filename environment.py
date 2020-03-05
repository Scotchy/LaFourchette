import numpy as np
from agents import BaseAgent
from utils import eating_hours

class Environment:

    def __init__(self, users, restaurants, agent: BaseAgent, seed=None):
        self.users = users
        self.n_users = len(users)
        self.restaurants = restaurants
        self.agent = agent
        self.seed = seed
        self.rd = np.random.RandomState(seed)
        self.day = 0
        self.hour = 0
        self.rewards = []
        self.max_rewards = []
        self.hungry_users = []
    
    def reset(self, specialities):
        seed = self.seed
        for user in self.users:
            user.generate_random_params(self.restaurants, specialities)
            if seed is not None:
                seed += 1
        for restaurant in self.restaurants:
            restaurant.generate_random_params(5)
            if seed is not None:
                seed += 1
        self.agent.init(seed)

    def step(self):
        
        # Piocher un utilisateur
        user_id = None
        available_restaurants = []
        while len(available_restaurants) == 0:
            user_id = None
            while user_id is None:
                self.tick()
                #print("day : {} hour : {}".format(self.day, eating_hours[self.hour]))
                self.get_hungry_users()
                user_id = self.new_user()

            # Filtrer des restaurants (pas trop loin, dans les prix)
            available_restaurants = self.filter_restaurants(user_id)
            if len(available_restaurants) == 0:
                self.users[user_id].update(-1)

        available_restaurants_id = [r.id for r in available_restaurants]
        user = self.users[user_id]
        # Faire choisir 5 restaus à l'agent
        context = np.array([np.concatenate((user.get_feature_vec(), restaurant.get_feature_vec())) for restaurant in self.restaurants])
        suggested_restaurants = [self.agent.get_action(context, user_id, available_restaurants_id) for i in range(1)]
        # Calculer le reward
        suggested_and_available = [self.restaurants[r] for r in suggested_restaurants]
        rest_id, table_id = user.chose_restaurant(suggested_and_available)

        reward, max_reward, grade = user.simulate_eating_at_restaurants(available_restaurants, rest_id, table_id)

        if rest_id is not None:
            restaurant = self.restaurants[rest_id]
            restaurant.give_grade(grade)
            restaurant.update_notoriety()
            
        user.update(grade)
        self.agent.update(context, suggested_restaurants[0], reward)

        self.rewards.append(reward)
        self.max_rewards.append(max_reward)
        if rest_id is not None:
            return user_id, rest_id, reward
        else:
            return user_id, suggested_restaurants[0], reward

    def get_cum_reward(self):
        return np.cumsum(self.rewards), np.cumsum(self.max_rewards)
    
    def tick(self):
        for restau in self.restaurants:
            restau.save_occupation()
            restau.free_tables()
        if self.hour == len(eating_hours) - 1:
            self.hour = 0
            self.day += 1
        else:
            self.hour += 1

    def filter_restaurants(self, user_id):
        user = self.users[user_id]
        lat, lon = user.lat, user.lon
        max_distance = user.max_distance
        possible_restaurants = []

        for restaurant in self.restaurants:
            distance = np.sqrt((lat - restaurant.lat) ** 2 + (lon - restaurant.lon) ** 2) * 5000
            # Remove restaurants that are too far, cannot welcome the amount of people, and are too expensive or too cheap
            if distance < 2 * max_distance and np.abs(restaurant.mean_price - user.price_appeatance) < 40:
                possible_restaurants.append(restaurant)
        return possible_restaurants

    def new_user(self):
        if len(self.hungry_users) == 0:
            return None
        return self.hungry_users.pop()
    
    def get_hungry_users(self):
        self.hungry_users = [i for i, u in enumerate(self.users) if u.next_eating_day == self.day and u.next_eating_hour == eating_hours[self.hour]]
        return len(self.hungry_users)
    