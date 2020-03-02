import numpy as np

class Environment:

    def __init__(self, users, restaurants, agent: Base, seed=None):
        self.users = users
        self.n_users = len(users)
        self.restaurants = restaurants
        self.seed = seed
        self.rd = np.random.RandomState
        self.day = 0
        self.hour = 0
        self.eating_hours = [11, 12, 13, 14, 18, 19, 20, 21, 22]
    
    def reset(self):
        seed = self.seed
        for user in self.users:
            user.generate_random_params(self.restaurants, seed)
            seed += 1
        for restaurant in self.restaurants:
            restaurant.generate_random_params(5, seed)
            seed += 1

    def step(self, day, next_hour=False):
        if next_hour:
            for restau in self.restaurants:
                restau.free_tables()
        
        # Piocher un utilisateur
        user_id = self.new_user()
        # Set le day
        # Filtrer des restaurants (pas trop loin, dans les prix)
        available_restaurants = self.filter_restaurants(user_id)
        # Faire choisir 5 restaus à l'agent
        # Calculer le reward
        r = ...
        # Calculer le max reward
        pass
    
    def filter_restaurants(self, user_id):
        user = self.users[user_id]
        lat, lon = user.lat, user.lon
        max_distance = user.max_distance
        possible_restaurants = []

        for restaurant in self.restaurants:
            distance = np.sqrt((lat - restaurant.lat) ** 2 + (lon - restaurant.lon) ** 2) * 50000
            # Remove restaurants that are too far, cannot welcome the amount of people, and are too expensive or too cheap
            if distance < 2 * max_distance and restaurant.can_welcome(user.n_people) and np.abs(restaurant.mean_price - user.price_appeatance) < 40:
                possible_restaurants.append(restaurant)
        return possible_restaurants

    def new_user(self):

        rand_ind = self.rd.randint(0, self.n_users)
        while not self.users[rand_ind].wanna_eat(self.hour):
            rand_ind = self.rd.randint(0, self.n_users)
        return rand_ind

    def get_reward(self, user, restaurant, commission):
        pass

    def get_rest_reward(self):
        pass
    
    def get_max_reward(self, user, restaurants):
        commissions = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]
        for restau in restaurants:
            for comm in commissions:
                # Combien le client aurait dépensé ? ==> A combien se serait élevée la commission ?
                # Est ce que le restaurant serait parti ?
                # 
                pass
        pass

    def get_max_commission(self, restaurant):
        possible_commissions = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]
        staying_probs = [restaurant.staying_prob(c) for c in possible_commissions]
        ind = np.sum()
    