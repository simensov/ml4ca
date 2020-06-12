#!/usr/bin/env python3

'''
This file contains the implementation of a simple feed forward neural network class using Keras.
It is implemented as a separate class to get easy access to different plotting, storage and loading functions to my own liking.
Note that the network solves a regression problem. Therefore, the last layer is linear, and the loss function is mean squared error. 
If using it for different purposes, such as binary classification, the last layer and loss function must be changed.

Example initialization:

nn = FFNeuralNetwork(10,2,20,3)

@author Simen Sem Oevereng, simensem@gmail.com, November 2019.
'''
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import max_norm
from keras.utils import plot_model
from keras.models import model_from_json, load_model

from numpy import sqrt
from ann_visualizer.visualize import ann_viz # Draws a regular neural network to plt. Not working with droput-layers etc.
import matplotlib.pyplot as plt

# TODO not used. Creates a leaky relu activation function
lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

class FFNeuralNetwork():
    '''
    Implementation of a Feedforward Neural Network for regression tasks, using:
        - Mean Squared Error as loss function.
        - Adam as optimizer.
        - Adjustable Dropout rates (use_dropout = False as standard for smaller networks due to underfitting problems).
        - Restrictable layer and bias weight magnitudes.
        - The same width of all hidden layers (this could however be changed manually)
    '''

    def __init__(self, 
                      input_dim = 3, 
                      num_hidden = 2, 
                      hidden_nodes = 3, 
                      output_dim = 6, 
                      use_dropout = False, 
                      dropout_rate = 0.25, 
                      restrict_norms = False, 
                      norm_max = 5.0,
                      loss = 'mean_squared_error',
                      activation = 'relu',
                      optimizer = 'adam',
                      metrics = []):

        self.input_dim      = input_dim
        self.num_hidden     = num_hidden
        self.hidden_nodes   = hidden_nodes
        self.output_dim     = output_dim
        self.use_dropout    = use_dropout
        self.do_rate        = dropout_rate
        self.restrict_norms = restrict_norms
        self.norm_max       = norm_max
        self.loss           = loss
        self.activation     = activation
        self.optimizer      = optimizer
        self.metrics        = metrics
        self.model          = self.nn_model()
        self.history        = {}

    def __repr__(self):
        return 'FFNN: {} inputs, {} hidden layers, {} neurons per layer, {} outputs'.format(self.input_dim, self.num_hidden, self.hidden_nodes, self.output_dim)

    def nn_model(self):
        '''
        Creates the neural network architecture
        '''

        # Notes: use droput and kernel maximization could avoid overfitting: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
        # he_normalization used with relu activation: https://arxiv.org/abs/1502.01852
        model = Sequential()

        # Initialize first layer with correct input layer
        model.add(Dropout(self.do_rate, input_shape=(self.input_dim,))) if self.use_dropout else None

        model.add(Dense(
                        units              = self.hidden_nodes,
                        input_dim          = self.input_dim,
                        kernel_initializer = 'glorot_uniform',
                        bias_initializer   = 'normal',
                        activation         = self.activation,
                        kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                        bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None),
                        name               = 'input_layer'))

        # Hidden layers
        for hidden_layer in range(self.num_hidden): 
            model.add(Dropout(self.do_rate)) if self.use_dropout else None
            model.add(Dense(
                             units              = self.hidden_nodes,
                             kernel_initializer = 'glorot_uniform',
                             bias_initializer   = 'normal',
                             activation         = self.activation,
                             kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                             bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None),
                             name               = 'hidden_layer_{}'.format(hidden_layer+1)))

        # Output layer
        model.add(Dropout(self.do_rate)) if self.use_dropout else None
        model.add(Dense(
                        units              = self.output_dim,
                        kernel_initializer = 'glorot_uniform', # he_normal, normal, glorot_uniform
                        activation         = 'linear',
                        kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                        bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None),
                        name               = 'output_layer'))

        # Compile model
        model.compile(
                      loss      = self.loss, # root_mean_squared_error, # 'mean_squared_error'
                      optimizer = self.optimizer, # adadelta, adam etc.
                      metrics   = self.metrics) # 'acc' does not give any meaning for regression problems.

        return model

    def model2png(self):
        plot_model(self.model, to_file='model.png')

    def visualize(self,title=''):
        ann_viz(self.model, title=title)

    def plotHistory(self,plot_validation=True):
        if self.history == 0:
            print('No history keras objects has been assigned')
        else:
            plt.figure()            
            # Plot training & validation loss values
            plt.subplot(111) # 211
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss']) if plot_validation else None
            plt.title('Mean Squared Error')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)

    def saveModel(self):
        '''
        Saves keras for later use. Needed for implementation in ROS.
        Saves model as a HDF5 file with:
            the architecture of the model, allowing to re-create the model
            the weights of the model
            the training configuration (loss, optimizer)
            the state of the optimizer, allowing to resume training exactly where you left off. 

        From: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        Needs h5py: http://docs.h5py.org/en/stable/build.html
        '''
        self.model.save('model.h5')
        print('Saved model')

    def loadModel(self):
        '''
        Loads previously compiled (and trained) model. Needed for implementation in ROS.
        From: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        Needs h5py: http://docs.h5py.org/en/stable/build.html
        '''
        self.model = load_model('model.h5')


# TODO move such utilities into another directory
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))