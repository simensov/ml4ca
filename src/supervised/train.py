#!/usr/bin/env python

# import importlib
# importlib.import_module('sl')

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.utils import model_to_dot
from keras.models import model_from_json
from keras.constraints import max_norm
from ann_visualizer.visualize import ann_viz
from tensorflow.keras.metrics import RootMeanSquaredError
import numpy as np

from sl import SupervisedTau
import matplotlib.pyplot as plt

st = SupervisedTau()
# st.loadData('dataset_train.npy')
st.loadData('dataset_train_fixedazi.npy')
# st.loadData('dataset_train_2121.npy')
st.loadData('dataset_train_1111.npy')


# st.generateData()
# st.displayData()
# print(st.maxMSE())

dataset = st.data



np.random.seed(0)

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

class FFNeuralNetwork():
    '''
    Implementation of a Feedforward Neural Network
    '''

    def __init__(self, input_dim = 3, num_hidden = 2, hidden_nodes = 3, output_dim = 6, use_dropout = False, dropout_rate = 0.25, restrict_norms = False, norm_max = 5.0):
        self.input_dim      = input_dim
        self.num_hidden     = num_hidden
        self.hidden_nodes   = hidden_nodes # list of length num hidden, representing number of nodes in each hidden layer
        self.output_dim     = output_dim
        self.use_dropout    = use_dropout
        self.do_rate        = dropout_rate
        self.restrict_norms = restrict_norms
        self.norm_max       = norm_max


        self.model = self.nn_model()
        self.history = 0

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
        model.add(Dropout(self.do_rate, input_shape=(self.input_dim,))) if self.use_dropout else None

        model.add(Dense(
                        units              = self.hidden_nodes,
                        input_dim          = self.input_dim,
                        kernel_initializer = 'normal',
                        activation         = 'relu',
                        kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                        bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None)))

        # Hidden layers
        for i in range(1,self.num_hidden):
            model.add(Dropout(self.do_rate)) if self.use_dropout else None
            model.add(Dense(
                             units              = self.hidden_nodes,
                             kernel_initializer = 'he_normal',
                             activation         = 'relu',
                             kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                             bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None)))

        # Output layer
        model.add(Dropout(self.do_rate)) if self.use_dropout else None
        model.add(Dense(
                        units              = self.output_dim,
                        kernel_initializer = 'he_normal',
                        kernel_constraint  = (max_norm(self.norm_max) if self.restrict_norms else None),
                        bias_constraint    = (max_norm(self.norm_max) if self.restrict_norms else None)))

        # Compile model
        model.compile(
                      loss      = 'mean_squared_error', # root_mean_squared_error, # 'mean_squared_error'
                      optimizer = 'rmsprop', # adadelta, adam etc.
                      metrics   = ["accuracy"]) #,RootMeanSquaredError(name='rmse')] )

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
            plt.subplot(211)
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)
            
            # Plot training & validation loss values
            plt.subplot(212)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            # plt.plot([st.maxMSE()]*len(self.history.history['val_loss'])) # Makes it very hard to visualize the loss
            plt.title('Model Mean Squared Error')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)

            # plt.subplot(224) # This looks super smooth, but test loss is lower, and I have no idea what is going on
            # plt.plot(self.history.history['rmse'])
            # plt.plot(self.history.history['val_rmse'])
            # plt.title('Model Root Mean Squared Error')
            # plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            # plt.legend(['Train', 'Test'], loc='upper left')

            # plt.show()

    def saveModel(self):
        # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def loadModel(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")



# evaluate model with standardized dataset

nn = FFNeuralNetwork(3,2,5,6,use_dropout=True,dropout_rate=0.3,restrict_norms=True,norm_max=5.0)

# nn.visualize() #  not able to visualize if these is a dropout layer
# plt.show()

# TODO standardize or normalize the training data (consider what happens to the output as well)

scaler = StandardScaler()
stdsc = scaler.fit(dataset)
dataset_scaled = scaler.transform(dataset)

# inverse_transform(dataset)
# Training data and test data
# X = dataset_scaled[:,0:3]
# Y = dataset_scaled[:,3:]


np.random.shuffle(dataset)
X = dataset[:,0:3]
Y = dataset[:,3:]
X = dataset_scaled[:,0:3]
Y = dataset_scaled[:,3:]

xtrain, xval, ytrain, yval = train_test_split(X,Y,test_size=0.33,shuffle=True)

# TODO test effect of dropout (0.2-0.5 rate, use on every layer)
# TODO test effect of weight constraints
# TODO use cross validation to select hyperparameters! more Dense layers (deeper), more hidden neurons (wider), droput rate, batch size
# TODO make sure to avoid underfitting first, attempting to overfit. Then, address overfitting
# TODO 

print('Traning model on {} datapoints'.format(X.shape[0]))
# nn = FFNeuralNetwork(3,2,2,6)
# nn.history = nn.model.fit(X,Y,validation_split = 0.1, epochs = 30, batch_size = 32, verbose = 0, shuffle = True)
# nn.plotHistory()


if True:
    print('Running cross-validation on nn architecture')
    best_loss = np.inf
    best_params = []
    best_history = nn.history
    for hidden_layers in range(1,4):
        for num_neurons in range(3,21,6):
            print('Training with {} hidden layer(s) having {} units each'.format(hidden_layers,num_neurons))
            nn = FFNeuralNetwork(3,hidden_layers,num_neurons,6,use_dropout=False,dropout_rate=0.2,restrict_norms=False,norm_max=5.0)
            nn.history = nn.model.fit(xtrain,ytrain,validation_data = (xval,yval), epochs = 30, batch_size = 32, verbose = 0, shuffle = True) # validation_split = 0.2 is overwritten by validation data
            print('Final training and validation accuracy: {} and {}'.format(nn.history.history['accuracy'][-1],nn.history.history['val_accuracy'][-1]))
            if best_loss > nn.history.history['val_loss'][-1]:
                best_loss = nn.history.history['val_loss'][-1]
                best_params = [hidden_layers,num_neurons,nn]

    print('Lowest final validation loss was found with {} hidden layer(s) and {} neurons each'.format(best_params[0],best_params[1]))
    best_params[2].plotHistory()


# Testing with 
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=nn.model, epochs=50, batch_size=32, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# TODO add test data
# score = nn.model.evaluate(X_test,Y_test,verbose=1)
# print(score)

plt.show()