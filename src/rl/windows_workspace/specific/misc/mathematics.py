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

def gaussian(val,mean=None,var=None):
    ''' Val, mean, variance all needs to be vectors of same dim; e.g. (3,1) or (3,). Return shape is the same '''
    if isinstance(val,list):
        val = np.array(val)

    if isinstance(mean,list):
        mean = np.array(mean)

    if isinstance(var,list):
        var = np.array(var)
        
    mean = np.zeros(val.shape) if (mean is None) or (mean.shape != val.shape) else mean
    var = np.ones(val.shape) if (var is None) or (var.shape != val.shape) else var
    return 1 / np.sqrt(2 * np.pi * var) * np.exp ( - 0.5 * ((val - mean) / np.sqrt(var))**2 ) # NB exp(-0.5) originally

def gaussian_like(val=[0],mean=[0],var=[1]):
    if isinstance(val,list):
        val = np.array(val)
    if isinstance(mean,list):
        mean = np.array(mean)
    if isinstance(var,list):
        var = np.array(var)

    return np.sqrt(2*np.pi*var) * gaussian(val,mean,var)


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