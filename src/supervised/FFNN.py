'''
This file contains the implementation of a simple feed forward neural network class.
It is implemented as a separate class to get easy access to different plotting, storage and loading functions to my own liking.
Note that the network solves a regression problem. Therefore, the last layer is linear, and the loss function is mean squared error. 
If using it for different purposes, such as binary classification, the last layer and loss function must be changed.

Example initialization:

nn = FFNeuralNetwork(10,2,20,3)

@author Simen Sem Oevereng, simensem@gmail.com, November 2019.
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import max_norm
from keras.utils import plot_model
from keras.models import model_from_json

from numpy import sqrt

from ann_visualizer.visualize import ann_viz # Draws a regular neural network to plt. Not working with droput-layers etc.

import matplotlib.pyplot as plt

class FFNeuralNetwork():
    '''
    Implementation of a Feedforward Neural Network for regression tasks, using:
        - Mean Squared Error as loss function.
        - Adam as optimizer.
        - Adjustable Dropout rates (use_dropout = False as standard for smaller networks due to underfitting problems).
        - Restrictable layer and bias weight magnitudes.
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
                      metrics = []):

        self.input_dim      = input_dim
        self.num_hidden     = num_hidden
        self.hidden_nodes   = hidden_nodes # list of length num hidden, representing number of nodes in each hidden layer
        self.output_dim     = output_dim
        self.use_dropout    = use_dropout
        self.do_rate        = dropout_rate
        self.restrict_norms = restrict_norms
        self.norm_max       = norm_max

        self.loss = loss
        self.metrics = metrics
        self.model = self.nn_model()
        self.history = {}

    def __repr__(self):
        return 'FFNN: {} inputs, {} hidden layers, {} neurons per layer, {} outputs'.format(self.input_dim, self.num_hidden, self.hidden_nodes, self.output_dim)

    def nn_model(self):
        '''
        Creates the neural network architecture
        '''

        # Notes: use droput and kernel maximization to avoid overfitting: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
        # he_normalization since using relu activation: https://arxiv.org/abs/1502.01852
        model = Sequential()

        # Initialize first layer with correct input layer
        # model.add(Dropout(self.do_rate, input_shape=(self.input_dim,))) if self.use_dropout else None

        model.add(Dense(
                        units              = self.hidden_nodes,
                        input_dim          = self.input_dim,
                        kernel_initializer = 'normal',
                        activation         = 'relu',
                        kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                        bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None)))

        # Hidden layers
        for _ in range(1,self.num_hidden):
            # model.add(Dropout(self.do_rate)) if self.use_dropout else None
            model.add(Dense(
                             units              = self.hidden_nodes,
                             kernel_initializer = 'he_normal',
                             activation         = 'relu',
                             kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                             bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None)))

        # Output layer
        # model.add(Dropout(self.do_rate)) if self.use_dropout else None
        model.add(Dense(
                        units              = self.output_dim,
                        kernel_initializer = 'he_normal', # he_normal, normal, glorot_uniform
                        kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                        bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None)))

        # Compile model
        model.compile(
                      loss      = self.loss, # root_mean_squared_error, # 'mean_squared_error'
                      optimizer = 'adam', # adadelta, adam etc.
                      metrics   = self.metrics) # 'acc' does not give any meaning for regression problems.

        return model

    def model2png(self):
        plot_model(self.model, to_file='model.png')

    def visualize(self,title=''):
        ann_viz(self.model, title=title)

    def plotHistory(self):
        if self.history == 0:
            print('No history kears objects has been assigned')
        else:
            plt.figure()            
            # Plot training & validation loss values
            plt.subplot(211)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Mean Squared Error')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)

            plt.subplot(212)
            plt.plot(sqrt(self.history.history['loss']))
            plt.plot(sqrt(self.history.history['val_loss']))
            plt.title('Root Mean Squared Error')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)
            # plt.show()

    def saveModel(self):
        '''
        Saves keras for later use. Needed for implementation in ROS.
        From: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        Needs h5py: http://docs.h5py.org/en/stable/build.html
        '''
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def loadModel(self):
        '''
        Loads previously compiled (and trained) model. Needed for implementation in ROS.
        From: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        Needs h5py: http://docs.h5py.org/en/stable/build.html
        '''
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")


# TODO move such utilities into another directory
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))