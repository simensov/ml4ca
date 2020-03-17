import numpy as np 

from keras.models import Model 
from keras.layers import Input, Dense
from keras import backend as K 
from keras.optimizers import Adam

from utils.mathematics import normal_dist

def clip_loss(self,A,pred_t):
    ''' Returns a loss function used when compiling actor network. Compared to the paper, which maximizes the objective, we minimize the loss, hence the minus in the end'''
    # TODO there should be two more penalities here: exploration/entropy (probably not needed due to gaussian noise) and 
    def loss(true_tp1, pred_tp1):
        variance = K.square(self.actor_noise)
        new_prob = normal_dist(pred_tp1,true_tp1,variance)
        old_prob = normal_dist(pred_t,true_tp1,variance)
        ratio = new_prob / (old_prob + np.random.uniform(-1,1)*(1e-10)) # 
        clipval = K.clip(ratio,1-self.actor_epsilon, 1+self.actor_epsilon) * A
        return -K.mean(K.minimum(ratio * A, clipval))


class TanhActor():

    def __init__(self,num_states,num_actions,action_bound):
        self.action_bound = action_bound # vector of largest magnitude of actions (NB: requires that the action bounds are symmetric around 0)

        inn = Input(shape = (num_states,))
        A = Input(shape = (1,)) # advantage
        old_pred = Input(shape=(num_actions,))
        x = Dense(self.layers[0], activation = 'tanh')(inn) # TODO test ReLus in the hidden layers?
        for i, hidden_nodes in enumerate(self.layer_dims):
            if i == 0: continue 
            x = Dense(hidden_nodes, activation = 'tanh')(x)
        x = Dense(self.actor_num_actions, activation='tanh')(x)
        model = Model(inputs=[inn, A, old_pred], outputs = [x])
        model.compile(optimizer=Adam(lr=self.actor_lr), loss = [self.clip_loss(A,old_pred)])
        model.summary()

        self.model = model

    def act(self,state):
        return self.model(state) * self.

class TestActor(object):

    def __init__(self,state_dim,action_dim,action_bound):
