#!/usr/bin/env python3

'''
https://keras.io/getting-started/faq/
Using TensorFlow 1.10 - Keras version 2.2.0. Latest releases of november 2019 was: TF 2.0 and Keras 2.3
  - pip install --ignore-installed --upgrade "Download URL" --user
  Link to find "Download URL" suitable for your specifications: https://github.com/lakshayg/tensorflow-build
  - pip install keras==2.2.0
  Link to find different TF and keras compatiblities: https://docs.floydhub.com/guides/environments/

This was done instead of just pip install tensorflow and pip install keras in order to optimize tensorflow for this computer's specific CPU for 3x calculation speeds

@author: Simen Oevereng, simensem@gmail.com, December 2019
'''
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

from SupervisedTau import SupervisedTau # Import the dataset generator, containing all tau,u,alpha datas
from FFNN import FFNeuralNetwork # Import FF neural network

'''
Create and suffle the data set (it has already been scaled in SupervisedTau)
'''
st = SupervisedTau()
st.loadData('dataset_train_3131.npy')
dataset = st.data

np.random.seed()
for _ in range(np.random.choice([3,5,7,9])):
    np.random.shuffle(dataset) # Shuffle dataset as it has been generated manually, and contains a lot of patterns

input_size = 9
label_size = 6
X = dataset[:,0:input_size]
Y = dataset[:,input_size:]

xtrain, xvaltmp, ytrain, yvaltmp = train_test_split(X,Y,test_size=0.2,shuffle=True)

# Create a testset, using train_test_split on the validation set. In such a way I can get a performance measure after the training, not only during validation checks
xval, xtest, yval, ytest = train_test_split(xvaltmp,yvaltmp,test_size=0.2,shuffle=True)

'''

TRAINING PHASE

'''

def wrapAngle(angle_deg):
    '''
    Wrap an angle in degrees between -180 and 180 deg
    '''
    return np.mod(angle_deg + 180.0, 360.0) - 180.0

def lookAtPredictions(xtest,ytest,nn):
    '''
    Used after training of a model in order to compare the predictions with the labels of the dataset for visualization of what's going on.
    It also calculates tau and tau_desired, and shows the magnitude of the error of those two vectors (elementwise).
    '''
    predictions = nn.model.predict(xtest)
    for idx,(u1,u2,u3,a1,a2,a3) in enumerate(predictions):
        
        scale_thrusters = 100
        scale_angles = np.pi # depends on angle normalization inside SupervisedTau
        u = np.array([u1,u2,u3]).T * scale_thrusters
        a = np.array([a1,a2,a3]).T * scale_angles
        
        sl = SupervisedTau()
        print('¤¤¤¤¤ Gains and angles vs. predicted')
        u1t,u2t,u3t,a1t,a2t,a3t = ytest[idx,:]
        realua = np.array([[u1t * scale_thrusters,u2t * scale_thrusters,u3t * scale_thrusters,wrapAngle(rad2deg(a1t * scale_angles)),wrapAngle(rad2deg(a2t* scale_angles)),wrapAngle(rad2deg(a3t* scale_angles))]]).T
        predua = np.array([[u1 * scale_thrusters,u2 * scale_thrusters,u3 * scale_thrusters,rad2deg(a1* scale_angles),rad2deg(a2* scale_angles),rad2deg(a3* scale_angles)]]).T
        print( np.hstack((realua,predua)) )

        print('¤¤¤¤¤ Difference in Tau vs. Predicted tau')
        tauscaled = xtest[idx:idx+1,-3:].T
        datascale = np.array([[54,69.2,76.9]]).T # calculated maximum taux, tauy, taup
        taureal = np.multiply(tauscaled,datascale) # elementwise multiplication

        print(taureal - sl.tau(a,u) )

        print('###########################################\n')

# Select scenario to run:
#   - 1: Train a simple neural network, used for actual training and saving models to be used in the simulator.
#   - 2: Hyperparameter testing, used to figure out which hyperparameters are the best.

scenario = 1

if scenario == 1:
    '''
    Implementation testing
    '''
    # Cross-validation gave 3-5 layers with 20-30 neurons as good numbers. 1 layer with 20 neurons gives good results as well for the simple case
    hidden_layers = 3
    num_neurons = 20
    nn = FFNeuralNetwork(input_size,hidden_layers,num_neurons,label_size,activation='relu', use_dropout=False,dropout_rate=0.3,restrict_norms=False,norm_max=10.0)
    print('Training nn with {} hidden layers, {} neurons on dataset with {} samples'.format(hidden_layers,num_neurons,xtrain.shape[0]))
    
    nn.model.summary()
    nn.history = nn.model.fit(xtrain,
                              ytrain,
                              validation_data = (xval,yval),
                              epochs          = 250,
                              batch_size      = 64,
                              verbose         = 0,
                              shuffle         = True) # validation_split = 0.2 is overwritten by validation data
    
    print('... Done!')
    nn.plotHistory()
    lookAtPredictions(xtest,ytest,nn)

    results = nn.model.evaluate(xtest,ytest) # Perform the final RMSE evaluation of the external test set, not used anywhere during training
    print('Test set RMSE: ',np.sqrt(results))

    print('Testing loading and saving + predictions of model')
    nn.saveModel()
    nn.loadModel()

    print('Displaying loaded model + predictions to confirm same performance as the original model')
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

# Show plots in the end
plt.show()