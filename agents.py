import numpy as np

def random_argmax(rd, l):
    return rd.choice(np.argwhere(l == l.max()).flatten())

class BaseAgent():

    def __init__(self, seed):
        self.rd = np.random.RandomState(seed=seed)

    def init(self, seed):
        self.rd = np.random.RandomState(seed=seed)

    def get_action(self, user, restaurants):
        pass
    
    def update(self, context, action, reward):
        pass

class RandomAgent(BaseAgent):

    def __init__(self, seed=None):
        super(RandomAgent, self).__init__(seed)

    def get_action(self, context, restaurants_id):
        n_restaurants = len(restaurants_id)
        if n_restaurants <= 0:
            return None
        return restaurants_id[self.rd.randint(0, n_restaurants)]
    
    

class EpsGreedyAgent(BaseAgent):
    def __init__(self, nb_arms, epsilon, seed=None):
        super(EpsGreedyAgent, self).__init__(seed)
        self.epsilon = epsilon
        self.nb_played = np.zeros(nb_arms)
        self.q = np.zeros(nb_arms)

    def get_action(self, user, restaurants):
        n_restaurants = len(restaurants)
        p = self.rd.rand()
        if p < self.epsilon:
            return restaurants[self.rd.randint(0, n_restaurants)]
        else:
            return restaurants[random_argmax(self.rd, self.q[restaurants])]
        
    def update(self, action, reward):
        self.nb_played[action] += 1
        self.q += (reward - self.q[action]) / self.nb_played[action]

class ContextualEpsilonGreedyAgent(BaseAgent):
    def __init__(self, nb_arms, context_size, lr=.1, epsilon=0, seed=None):
        super(ContextualEpsilonGreedyAgent, self).__init__(seed)
        self.nb_arms = nb_arms
        self.p = context_size
        self.lr = lr
        self.epsilon = epsilon
        self.beta = np.zeros((nb_arms, self.p)) 
        self.nb_played = np.zeros(nb_arms)
        self.pred_reward = np.zeros(nb_arms)
        
    def get_action(self, context, user, restaurants_id):
        
        if self.rd.random() < self.epsilon:
            action = self.rd.randint(self.nb_arms)
        else:
            pred_reward = np.einsum('ij,ij->i', context, self.beta)
            action = restaurants_id[random_argmax(self.rd, pred_reward[restaurants_id])]
        return action
        
    def update(self, context, action, reward):
        """ Simple gradient descent. """
        self.nb_played[action] += 1
        grad = - context[action] * (reward - context[action].dot(self.beta[action]))
        self.beta[action] = self.beta[action] - self.lr/self.nb_played[action] * grad

class UCBAgent(BaseAgent):
    def __init__(self, nb_arms, c=2., seed=None):
        super(UCBAgent, self).__init__(seed)
        self.q = np.zeros(nb_arms)
        self.nb_played = np.zeros(nb_arms)
        self.c = c
        self.t = 0
    
    def get_action(self, context, user, restaurants):
        if 0 in self.nb_played:
            action = self.rd.choice(np.where(self.nb_played==0)[0])
        else:
            ar = self.q + np.sqrt(self.c * np.log(self.t) / self.nb_played)
            action = restaurants[np.argmax(ar[restaurants])]
        return action
    
    def update(self, action, reward):
        self.t += 1
        self.nb_played[action] += 1
        self.q[action] += (reward - self.q[action])/self.nb_played[action]


from keras.layers import Embedding, Flatten, Dense, Dropout,Input
from keras.layers import Dot
from keras.models import Model

class DeepAgent(BaseAgent):

    def __init__(self, n_restaurants, n_users, epsilon, embedding_size=20, seed=None):
        super(DeepAgent, self).__init__(seed)
        self.n_users = n_users
        self.n_restaurants = n_restaurants
        self.embedding_size = embedding_size
        self.eps = epsilon
        self.current_user = None
        self.init()
        
    def init(self, seed=None):
        self.history = []
        
        user_inputs = Input(shape=(1,))
        restau_inputs = Input(shape=(1,))
        
        user_embedding = Embedding(output_dim=self.embedding_size,
                                        input_dim=self.n_users,
                                        input_length=1,
                                        name='user_embedding')
        restaurant_embedding = Embedding(output_dim=self.embedding_size,
                                        input_dim=self.n_restaurants,
                                        input_length=1,
                                        name='restaurant_embedding')
        
        user_vecs = Flatten()(user_embedding(user_inputs))
        restaurant_vecs = Flatten()(restaurant_embedding(restau_inputs))
        
        final_layer = Dot(axes=1)([user_vecs, restaurant_vecs])
        
        self.model = Model(inputs=[user_inputs, restau_inputs], outputs=[final_layer])
        self.model.compile(optimizer="adam", loss='mae')

    def get_action(self, context, user, restaurants):
        exp_reward = self.model.predict([ [user] * len(restaurants), restaurants])
        return restaurants[random_argmax(self.rd, exp_reward.T[0])]
        
    
    def fit(self, history):
        X1 = history[0,:]
        X2 = history[1,:]
        Y = history[2,:]
        self.model.fit([X1,X2], Y, verbose=0)

    def update(self, context, action, reward):

        pass