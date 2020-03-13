
import numpy as np 

from keras.models import Model 
from keras.layers import Input, Dense
from keras import backend as K 
from keras.optimizers import Adam

clip_epsilon = 0.2
exploration_noise = 1.0
gamma = 0.99
buffer_size = 2048

CONTINUOUS    = True
EPISODES      = 100000
LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS        = 10
NOISE         = 1.0 # Exploration noise
GAMMA         = 0.99
BUFFER_SIZE   = 2048
BATCH_SIZE    = 256
NUM_ACTIONS   = 4
NUM_STATE     = 8
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
ENTROPY_LOSS  = 5e-3
LR            = 1e-4  # Lower lr stabilises training greatly

def normal_dist(val,mean,var):
    ''' https://en.wikipedia.org/wiki/Normal_distribution '''
    return 1 / K.sqrt(2* np.pi * var) * K.exp( -0.5 * K.square( (val - mean) / var) )

class PPO(object):
    # TODO use my own FFNN class for this

    def __init__(self,
                 num_states  = 6,
                 layers      = (64,64),
                 critic_lr   = 1e-4,
                 actor_noise = 1.0,
                 actor_lr    = 1e-4,
                 actor_clip  = 0.15,
                 actor_num_actions = 5):

        self.num_states = num_states
        self.layer_dims = layers
        self.lr_c = critic_lr
        self.critic_params = [layers, critic_lr] # TODO might clean up init
        self.critic = self.build_critic()
        self.actor_params = [layers, actor_lr, actor_noise] # TODO might clean up init
        self.actor_noise = actor_noise
        self.actor_lr = actor_lr
        self.actor_epsilon = actor_clip
        self.actor_num_actions = actor_num_actions
        self.actor = self.build_actor()

    def build_critic(self):
        inn = Input(shape = (self.num_states,))
        x = Dense(self.layer_dims[0], activation = 'tanh')(inn)
        for i, hidden_nodes in enumerate(self.layer_dims):
            if i == 0: continue
            x = Dense(hidden_nodes,activation='tanh')(x)
        x = Dense(1)(x)
        model = Model(inputs = [inn], outputs=[x]) # TODO replace with sequential from FFNN?
        model.compile(optimizer = Adam(lr = self.lr_c), loss = 'mse')
        return model

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

    def build_actor(self):
        inn = Input(shape = (self.num_states,))
        A = Input(shape = (1,)) # advantage
        old_pred = Input(shape=(self.actor_num_actions,))
        x = Dense(self.layers[0], activation = 'tanh')(inn)
        for i, hidden_nodes in enumerate(self.layer_dims):
            if i == 0: continue 
            x = Dense(hidden_nodes, activation = 'tanh')(x)
        x = Dense(self.actor_num_actions, activation='tanh')(x)
        model = Model(inputs=[inn, A, old_pred], outputs = [x])
        model.compile(optimizer=Adam(lr=self.actor_lr), loss = [self.clip_loss(A,old_pred)])
        model.summary()
        return model

    
