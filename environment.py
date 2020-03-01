class Environment:

    def __init__(self, users, restaurants, seed=None):
        self.users = users
        self.restaurants = restaurants
        self.seed = seed
    
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
        # Set le day
        # Filtrer des restaurants (pas trop loin, dans les prix)
        # Faire choisir 5 restaus à l'agent
        # Calculer le reward
        # Calculer le max reward
        pass

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
    