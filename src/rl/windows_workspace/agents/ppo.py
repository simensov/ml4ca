
import numpy as np 

from keras.models import Model 
from keras.layers import Input, Dense
from keras import backend as K 
from keras.optimizers import Adam

from utils.mathematics import normal_dist, clip_loss

EPOCHS_ACTOR  = 1 
EPOCHS_CRITIC = 10
BUFFER_SIZE   = 2048
BATCH_SIZE    = 256
ENTROPY_LOSS  = 5e-3
# Inspired by structure in https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py

class PPO(object):
    ''' A PPO agent with continous action space, using a neural net function approximator and tanh activations on output '''
    # TODO use my own FFNN class for this # TODO use Tensorboard as writer to store training info

    def __init__(self,
                 num_states  = 6,
                 num_actions = 5,
                 layers      = (64,64),
                 critic_lr   = 1e-3,
                 actor_noise = 1.0,
                 actor_lr    = 1e-3, # low lr could stabilize training
                 actor_clip  = 0.15
                 ):

        self.num_states    = num_states
        self.layer_dims    = layers
        self.critic_lr     = critic_lr
        self.critic        = self.build_critic()
        self.actor_noise   = actor_noise
        self.actor_lr      = actor_lr
        self.actor_epsilon = actor_clip
        self.num_actions   = num_actions
        self.actor         = self.build_actor()

        self.dummy_val, self.dummy_act = np.zeros((1,1)), np.zeros((1,num_actions))

    def build_critic(self):
        inn = Input(shape = (self.num_states,),name='critic_input')
        x = Dense(self.layer_dims[0], activation = 'tanh')(inn)
        for i, hidden_nodes in enumerate(self.layer_dims):
            if i == 0: continue
            x = Dense(hidden_nodes,activation='tanh')(x)
        x = Dense(1)(x)
        m = Model(inputs = [inn], outputs=[x])
        m.compile(optimizer = Adam(lr = self.critic_lr), loss = 'mse')
        # m.summary()
        return m

    def V(self,state):
        return self.critic.predict([state.reshape(1,self.num_states)]) # TODO a normal forward pass could be beneficial here!

    def clip_loss_fn(self,A,pred_t):
        ''' Returns a loss function used when compiling actor network. Compared to the paper, which maximizes the objective, we minimize the loss, hence the minus in the end'''
        # TODO there should be two more penalities here: exploration/entropy (probably not needed due to gaussian noise) and value function
        # -> total_loss = critic_coeff * K.mean(K.square(rewards - values)) + actor_loss - entropy_beta * K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        def loss(true_tp1, pred_tp1):
            variance = K.square(self.actor_noise)
            new_prob = normal_dist(pred_tp1,true_tp1,variance)
            old_prob = normal_dist(pred_t,true_tp1,variance)
            # TODO represent these as logs? Seems better for computational complexity
            ratio = K.exp( K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10) )
            # ratio = new_prob / (old_prob + 1e-10) 
            clipval = K.clip(ratio,1-self.actor_epsilon, 1+self.actor_epsilon) * A
            return -K.mean(K.minimum(ratio * A, clipval)) # maximize surrogate: minimize negative surrogate

        return loss

    def build_actor(self):
        inn  = Input(shape = (self.num_states,), name = 'actor_input')
        A    = Input(shape = (1,), name = 'actor_adv') # advantage
        prev = Input(shape = (self.num_actions,), name = 'actor_oldpred') # old prediction
        x    = Dense(self.layer_dims[0], activation = 'tanh')(inn)
        for i, hidden_nodes in enumerate(self.layer_dims):
            if i == 0: continue 
            x = Dense(hidden_nodes, kernel_initializer = 'glorot_uniform', bias_initializer = 'normal', activation = 'tanh')(x)
        
        x = Dense(self.num_actions, kernel_initializer = 'glorot_uniform', bias_initializer = 'normal', activation = 'tanh')(x)
        m = Model(inputs=[inn, A, prev], outputs = [x]) # inputs like these in order to pass advantage and previous actions to loss function
        m.compile(optimizer=Adam(lr=self.actor_lr), loss = self.clip_loss_fn(A,prev))
        # model.summary()
        return m

    def act(self,obs,epsilon=0.1):
        pred = self.actor.predict([obs.reshape(1,self.num_states), self.dummy_val, self.dummy_act]) # forward pass TODO predict used earlier        
        
        if np.random.random() < epsilon:
            act = pred + np.random.normal(loc=0, scale=self.actor_noise, size=pred[0].shape) # before it said pred[0]
        else:
            act = pred

        return act.reshape((self.num_actions,)), pred.reshape((self.num_actions,))

    # TODO collection of batch happens outside, see get_batch() from https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py 
    def update(self, batch):
        ''' o: observations - a: actions under new policy - p: predicted actions from old policy - r: discounted rewards (r + gamma * V) '''
        o, a, p, r = batch
        o, a, p, r = o[:BATCH_SIZE], a[:BATCH_SIZE], p[:BATCH_SIZE], r[:BATCH_SIZE]
        old_prediction = p
        pred_values    = self.critic.predict(o) # self.critic.predict(o) # forward pass
        advantage      = r - pred_values # standardize? 
        actor_loss     = self.actor.fit([o, advantage, old_prediction], [a], batch_size=BATCH_SIZE, shuffle=False, epochs=EPOCHS_ACTOR, verbose=False)
        critic_loss    = self.critic.fit([o], [r], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS_CRITIC, verbose=False)
        # remember that the "loss" is just an negative of the performance measurement under the current policy.
        # Once a single SGD step is taken, there is no connection to the performance of the current policy anymore
        # However, the value function itself just tries to find the optimal estimate of the average future reward, so it can be updated more often

        return advantage, actor_loss, critic_loss

    def load(self,filename='model.h5'):
        pass # TODO see src/sl/SupervisedTau.py

    def save(self,filename='model.h5'):
        pass # TODO see src/sl/SupervisedTau.py

    def __str__(self):
        self.critic.summary()
        self.actor.summary()
        return ''


if __name__ == '__main__':
    agent = PPO()

        
    '''

    # alternative loss, including entropy and value function loss!
    def ppo_loss(oldpolicy_probs, advantages, rewards, values):
        def loss(y_true, y_pred):
            newpolicy_probs = y_pred
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            critic_loss = K.mean(K.square(rewards - values))
            total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
                -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
            return total_loss

        return loss

    # automate advantage calculations!
    def get_advantages(values, masks, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    '''