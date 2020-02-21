'''
Builds on https://www.idi.ntnu.no/emner/it3105/materials/code/splitgd.py
Modified by Simen Øvereng
A wrapper around a keras model allowing for modifying gradients within tensorflow.
-> Instead of calling keras_model.fit, call CustomModel.fit, which calculates and modifies new gradients, and then applies them with the model's optimizer 
'''

import math
import tensorflow as tf # using tf here instead of tf due to local crash with the tf-package in opt/bin/tf, coming from ROS
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class CustomModel():

    def __init__(self,keras_model=None,input_size=16, layer_sizes = (20,1), decay=0.9,gamma=0.95, alpha=0.05):
        self.decay = decay
        self.gamma = gamma
        self.alpha = alpha
        self.model = keras_model if keras_model else self.createModel(input_size,layer_sizes)
        self.e     = self.resetEligibilities()

    def createModel(self,input_size,layer_sizes):
        ''' Creates a keras model with input and hidden layers according to layer sizes'''

        if not layer_sizes:
            raise Exception('Tuple of layer sizes has not been properly defined')

        model = Sequential(name='feed_forward_neural_network_eligibility_trace')
        model.add(Dense(layer_sizes[0], input_shape = (input_size,),name='hidden_layer_0'))

        for i, size in enumerate(layer_sizes):
            if i > 0 and i < len(layer_sizes) - 1:
                model.add(Dense(size, activation = 'relu', name='hidden_layer_{}'.format(i)))

        model.add(Dense(1, activation = 'linear',name='output')) 
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.alpha), metrics = ['mae'])
        return model


    def resetEligibilities(self):
        '''
        Reset eligibility matrix as a list of matrices and vectors. 
        The structure of the list is the following: [layer1_weights,layer1_biases,layer2_weights,layer2_biases,...] 
        '''

        network_shape = [(layer.input_shape[1], layer.output_shape[1]) for layer in self.model.layers]
        arr = []
        for rows, cols in network_shape:
            arr.append(np.zeros((rows,cols)))
            arr.append(np.zeros((cols)))
        
        self.e = arr
        return arr

    def modify_gradients(self, gradients, td_error = 0):
        '''
        Adjust gradients of the neural network with eligibility traces
        Gradients from argument is dL/dw_i, which gives dV/dw_i = -dL/dw / (2*delta) from Keith Downing's note 
        '''
       
        new_grad = gradients[:] # Copy list - to modify and return later

        # Update eligibility according to e(w_i) <- e(w_i) + dV/dw_i
        # Then, since Adam updates the weights in apply_gradients with w <- w - alpha * gradient,
        # and we have calculated according to w <- w + alpha * delta * e, we must return -delta * e
        for i, grad in enumerate(gradients):
            dVdw         = -grad.numpy() / (2*td_error)
            self.e[i]    = np.add(self.e[i], dVdw ) # update eligibility - element-wise addition
            mod_gradient = -td_error * self.e[i] # element-wise multiplication, reversing sign for Adam
            new_grad[i]  = tf.reshape(mod_gradient, grad.shape) # cast and reshape into Tensorflow specific Tensor-variable

        # Decay the eligibility
        self.e = [self.gamma * self.decay * eligibility for eligibility in self.e]

        return new_grad

    def gen_loss(self,features,targets,avg=False):
        ''' Returns a tensor of losses'''
        predictions = self.model(features)  # feed-forward pass to produce outputs/predictions
        return self.model.loss_functions[0](targets,predictions) # return loss variable

    def fit(self, features, targets, td_error, epochs=1, mbs=1,vfrac=0.1,verbose=True):
        ''' Adjusts the weights of the network '''
        params = self.model.trainable_weights # get list with all weights and biases as a list from input layer to output layer

        with tf.GradientTape() as tape:  # the tape from tf2.0 watches all operations and enable automatic differentiation
            loss      = self.gen_loss(features,targets,avg=False)
            gradients = tape.gradient(loss,params) # calculates gradient of loss wrt all training parameters
            gradients = self.modify_gradients(gradients,td_error) # modifies gradients according to eligibility traces
            self.model.optimizer.apply_gradients(zip(gradients,params)) # adjust the weights of the model