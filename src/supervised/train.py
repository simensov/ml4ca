#!/usr/bin/env python3

# https://keras.io/getting-started/faq/
# Using TensorFlow 1.10 - Keras version 2.2.0. Latest releases of november 2019 was: TF 2.0 and Keras 2.3
#   - pip install --ignore-installed --upgrade "Download URL" --user
#   Link to find "Download URL" suitable for your specifications: https://github.com/lakshayg/tensorflow-build
#   - pip install keras==2.2.0
#   Link to find different TF and keras compatiblities: https://docs.floydhub.com/guides/environments/
#
# This was done instead of just pip install tensorflow and pip install keras in order to optimize tensorflow for this computer's specific CPU for 3x calculation speeds

import numpy as np
from numpy import rad2deg
np.set_printoptions(precision=3) # print floats as decimals with 3 zeros

from pandas import read_csv
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import time

import tensorflow as tf
lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)


# Import the dataset generator, containing all tau,u,alpha datas
from SupervisedTau import SupervisedTau
from FFNN import FFNeuralNetwork

'''
Create, shuffle and scale the dataset
'''
st = SupervisedTau()
st.loadData('dataset_train_5151.npy')
dataset = st.data

if False:
    # TODO standardize or normalize the training data (consider what happens to the output as well)
    scaler         = StandardScaler()
    stdsc          = scaler.fit(dataset)
    dataset_scaled = scaler.transform(dataset)
    np.random.shuffle(dataset)
    X = dataset_scaled[:,0:3]
    Y = dataset_scaled[:,3:]

np.random.seed()
for _ in range(np.random.choice([3,5,7,9])):
    np.random.shuffle(dataset) # Shuffle dataset as it has been generated manually, and contains a lot of patterns

input_size = 9
label_size = 6
X = dataset[:,0:input_size]
Y = dataset[:,input_size:]

# print(np.max(X[:,6]),np.min(X[:,6]),np.max(X[:,7]),np.min(X[:,7]),np.max(X[:,8]),np.min(X[:,8]) )

xtrain, xvaltmp, ytrain, yvaltmp = train_test_split(X,Y,test_size=0.2,shuffle=True)
# TODO create a testset, using train_test_split on the validation set.
xval, xtest, yval, ytest = train_test_split(xvaltmp,yvaltmp,test_size=0.2,shuffle=True)



'''

TRAINING PHASE

'''

def lookAtPredictions(xtest,ytest,nn):
    predictions = nn.model.predict(xtest)
    for idx,(u1,u2,u3,a1,a2,a3) in enumerate(predictions):
        # u = np.array([u1,u2,u3]).T # * 100
        # a = np.array([a1,a2,a3]).T # 2*np.arcsin()

        u = np.array([u1,u2,u3]).T * 100
        a = np.array([a1,a2,a3]).T * np.pi
        
        sl = SupervisedTau()
        print('¤¤¤¤¤ Gains and angles vs. predicted')
        u1t,u2t,u3t,a1t,a2t,a3t = ytest[idx,:]
        realua = np.array([[u1t * 100,u2t * 100,u3t * 100,rad2deg(a1t* np.pi),rad2deg(a2t* np.pi),rad2deg(a3t* np.pi)]]).T
        predua = np.array([[u1 * 100,u2 * 100,u3 * 100,rad2deg(a1* np.pi),rad2deg(a2* np.pi),rad2deg(a3* np.pi)]]).T
        print( np.hstack((realua,predua)) )

        print('¤¤¤¤¤ Difference in Tau vs. Predicted tau')
        tauscaled = xtest[idx:idx+1,-3:].T
        datascale = np.array([[54,69.2,76.9]]).T # calculated maximum taux, tauy, taup
        taureal = np.multiply(tauscaled,datascale) # elementwise multiplication

        print(taureal - sl.tau(a,u) )

        print('###########################################\n')


scenario = 1

if scenario == 1:
    '''
    Implementation testing
    '''
    # xval gave 5, 30 as a good number. 1, 20 gives good results as well
    hidden_layers = 5
    num_neurons = 30
    nn = FFNeuralNetwork(input_size,hidden_layers,num_neurons,label_size,activation='relu', use_dropout=False,dropout_rate=0.3,restrict_norms=False,norm_max=10.0)
    print('Training nn with {} hidden layers, {} neurons on dataset with {} samples'.format(hidden_layers,num_neurons,xtrain.shape[0]))
    nn.model.summary()
    
    nn.history = nn.model.fit(xtrain,
                              ytrain,
                              validation_data = (xval,yval),
                              epochs          = 20,
                              batch_size      = 64,
                              verbose         = 0,
                              shuffle         = True) # validation_split = 0.2 is overwritten by validation data
    
    print('... Done!')
    nn.plotHistory()
    lookAtPredictions(xtest,ytest,nn)

    results = nn.model.evaluate(xtest,ytest)
    print('Test set RMSE: ',np.sqrt(results))

    print('Testing loading and saving + predictions of model')
    nn.saveModel()
    nn.loadModel()

    nn.model.summary()

    results = nn.model.evaluate(xtest,ytest)
    print('Test set RMSE: ',np.sqrt(results))
    

elif scenario == 2:
    '''
    Hyperparameter selection
    '''
    print('Running cross-validation on nn architecture')
    best_loss    = np.inf
    best_params  = []
    best_history = {}
    total_time   = 0
    for hidden_layers in range(1,5,2):
        for num_neurons in range(10,21,5):
            print('Training with {} hidden layer(s) having {} units each'.format(hidden_layers,num_neurons))

            nn = FFNeuralNetwork(input_size,hidden_layers,num_neurons,label_size,use_dropout=False,dropout_rate=0.2,restrict_norms=False,norm_max=5.0)
            
            stime = time.time() # start time
            
            nn.history = nn.model.fit(xtrain,ytrain,validation_data = (xval,yval), epochs = 20, batch_size = 32, verbose = 0, shuffle = True) # validation_split = 0.2 is overwritten by validation data
            
            endtime     = time.time() - stime # training time
            total_time += endtime # total time

            print('Resulting training loss: {} and validation loss: {}, trained in {:.2f} sec'.format(nn.history.history['loss'][-1],nn.history.history['val_loss'][-1], endtime))
            
            results = nn.model.evaluate(xtest,ytest)
            if results < best_loss:
                best_loss   = results
                best_params = [hidden_layers,num_neurons,nn]
    
    lookAtPredictions(xtest,ytest,best_params[2])
    
    print('Lowest final validation loss was found with {} hidden layer(s) and {} neurons each. Total time: {:.2f}'.format(best_params[0],best_params[1],total_time))
    
    best_params[2].plotHistory()


else:
    print('Do nothing')

    


# TODO add test data
# TODO test effect of dropout (0.2-0.5 rate, use on every layer)
# TODO test effect of weight constraints
# TODO use cross validation to select hyperparameters! more Dense layers (deeper), more hidden neurons (wider), droput rate, batch size
# TODO make sure to avoid underfitting first, attempting to overfit. Then, address overfitting
# TODO is there any point in using model.fit(x,y,classweight = classweight,...), as a dict, to use predefined weights?
# TODO consider early stopping: keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False) # https://keras.io/callbacks/
# score = nn.model.evaluate(X_test,Y_test,verbose=1)
# print(score)

plt.show()

# IDEA represent the us and as along the real number line, representing a unique combination of the thrusts.