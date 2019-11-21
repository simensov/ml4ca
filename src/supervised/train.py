#!/usr/bin/env python3

# Using TensorFlow 1.10 - Keras version 2.2.0. Latest releases of november 2019 was: TF 2.0 and Keras 2.3
#   - pip install --ignore-installed --upgrade "Download URL" --user
#   Link to find "Download URL" suitable for your specifications: https://github.com/lakshayg/tensorflow-build
#   - pip install keras==2.2.0
#   Link to find different TF and keras compatiblities: https://docs.floydhub.com/guides/environments/
#
# This was done instead of just pip install tensorflow and pip install keras in order to optimize tensorflow for this computer's specific CPU for 3x calculation speeds

import numpy as np

from pandas import read_csv
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import time


# Import the dataset generator, containing all tau,u,alpha datas
from sl import SupervisedTau
from FFNN import FFNeuralNetwork


'''
Create, shuffle and scale the dataset
'''
st = SupervisedTau()
# st.loadData('dataset_train.npy')
# st.loadData('dataset_train_fixedazi.npy')
#st.loadData('dataset_train_2121.npy')
st.loadData('dataset_train_1111.npy')
#st.loadData('dataset_train_55.npy')

dataset = st.data

if False:
    # TODO standardize or normalize the training data (consider what happens to the output as well)
    scaler = StandardScaler()
    stdsc = scaler.fit(dataset)
    dataset_scaled = scaler.transform(dataset)
    np.random.shuffle(dataset)
    X = dataset_scaled[:,0:3]
    Y = dataset_scaled[:,3:]

np.random.seed(0)
for _ in range(5):
    np.random.shuffle(dataset) # Shuffle dataset as it has been generated manually, and contains a lot of patterns

X = dataset[:,0:3]
Y = dataset[:,3:]

xtrain, xvaltmp, ytrain, yvaltmp = train_test_split(X,Y,test_size=0.2,shuffle=True)
# TODO create a testset, using train_test_split on the validation set.
xval, xtest, yval, ytest = train_test_split(xvaltmp,yvaltmp,test_size=0.2,shuffle=True)

'''
Hyperparameter selection
'''
if True:
    nn = FFNeuralNetwork(3,3,15,6,use_dropout=False,dropout_rate=0.3,restrict_norms=False,norm_max=5.0)
    print('Training nn...')
    nn.history = nn.model.fit(xtrain,ytrain,
                          validation_data = (xval,yval), 
                          epochs = 40, 
                          batch_size = 32, 
                          verbose = 0, 
                          shuffle = True) # validation_split = 0.2 is overwritten by validation data
    print('... Done!')
    nn.plotHistory()
    results = nn.model.evaluate(xtest,ytest)
    print('Test results: {}'.format(results))

    predictions = nn.model.predict(xtest)
    for idx,(u1,u2,u3,a1,a2,a3) in enumerate(predictions):
        u = 100 * np.array([u1,u2,u3]).T
        a = 2 * np.arcsin(np.array([a1,a2,a3]).T)
        print(u.T,a.T*180/np.pi)
        # sl = SupervisedTau()
        # print(xtest[idx,:],sl.tau(a,u).T)


else:
    print('Running cross-validation on nn architecture')
    best_loss = np.inf
    best_params = []
    best_history = {}
    total_time = 0
    for hidden_layers in range(2,5):
        for num_neurons in range(10,21,5):
            print('Training with {} hidden layer(s) having {} units each'.format(hidden_layers,num_neurons))

            nn = FFNeuralNetwork(3,hidden_layers,num_neurons,6,use_dropout=True,dropout_rate=0.2,restrict_norms=True,norm_max=5.0)
            
            stime = time.time() # start time 
            
            nn.history = nn.model.fit(xtrain,ytrain,validation_data = (xval,yval), epochs = 30, batch_size = 32, verbose = 0, shuffle = True) # validation_split = 0.2 is overwritten by validation data
            
            endtime = time.time() - stime # training time
            total_time += endtime # total time

            print('Resulting training loss: {} and validation loss: {}, trained in {:.2f} sec'.format(nn.history.history['loss'][-1],nn.history.history['val_loss'][-1], endtime))
            
            if best_loss > nn.history.history['val_loss'][-1]:
                best_loss = nn.history.history['val_loss'][-1]
                best_params = [hidden_layers,num_neurons,nn]

    print('Lowest final validation loss was found with {} hidden layer(s) and {} neurons each. Total time: {:.2f}'.format(best_params[0],best_params[1],totaltime))
    best_params[2].plotHistory()


# TODO add test data
# TODO test effect of dropout (0.2-0.5 rate, use on every layer)
# TODO test effect of weight constraints
# TODO use cross validation to select hyperparameters! more Dense layers (deeper), more hidden neurons (wider), droput rate, batch size
# TODO make sure to avoid underfitting first, attempting to overfit. Then, address overfitting
# TODO is there any point in using model.fit(x,y,classweight = classweight,...), as a dict, to use predefined weights?
# TODO consider early stopping: keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
# score = nn.model.evaluate(X_test,Y_test,verbose=1)
# print(score)

plt.show()

# IDEA represent the us and as along the real number line, representing a unique combination of the thrusts.