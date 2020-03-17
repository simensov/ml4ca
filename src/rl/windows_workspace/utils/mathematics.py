import numpy as np 
from keras import backend as K

'''
General
'''
def rotation_matrix(a:float):
    return np.array([ [np.cos(a), -np.sin(a)],
                      [np.sin(a),  np.cos(a)]]) 

def clip(val,low,high) -> float:
    return max(min(val, high), low)

def wrap_angle(angle,deg = True):
    ''' Wrap an angle between -180 and 180 deg. deg == True means degrees, == False means radians'''
    ref = 180.0 if deg else np.pi
    return np.mod(angle + ref, 2*ref) - ref

'''
TF specific
'''
def normal_dist(val,mean,var):
    ''' https://en.wikipedia.org/wiki/Normal_distribution '''
    return 1 / K.sqrt(2* np.pi * var) * K.exp( -0.5 * K.square( (val - mean) / var) )


def clip_loss(A,pred_t):
    ''' Returns a loss function used when compiling actor network. Compared to the paper, which maximizes the objective, we minimize the loss, hence the minus in the end'''
    # TODO there should be two more penalities here: exploration/entropy (probably not needed due to gaussian noise) and value function
    actor_noise = 1.0
    clipping_epsilon = 0.2
    def loss(true_tp1, pred_tp1):
        variance = K.square(actor_noise)
        new_prob = normal_dist(pred_tp1,true_tp1,variance)
        old_prob = normal_dist(pred_t,true_tp1,variance)
        ratio = new_prob / (old_prob + np.random.uniform(-1,1)*(1e-10)) # 
        clipval = K.clip(ratio,1-clipping_epsilon, 1+clipping_epsilon) * A
        return -K.mean(K.minimum(ratio * A, clipval))