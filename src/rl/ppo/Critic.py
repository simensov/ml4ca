#!/home/revolt/Documents/Simen/A.NTNU/projectthesis/ml4ta/src/rl/ppo/venv/bin/python

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K 

import CustomModel

class Critic():
    def __init__(self,decay=0.9,alpha=0.05,gamma=0.95,is_tabular=True,input_size=16,layer_sizes=(20,1)):
        self.is_tabular     = is_tabular
        self.alpha          = alpha if is_tabular else 0.0005
        self.gamma          = gamma
        self.value_function = dict() if is_tabular else CustomModel( self.create_model(input_size,layer_sizes), decay=decay,gamma=gamma)
        self.e              = dict()
        self.decay          = decay
    
    def create_model(self,input_size=16,layer_sizes=(20,1)):
        ''' Creates a keras model with input and hidden layers according to layer sizes'''

        if not layer_sizes:
            raise Exception('Tuple of layer sizes has not been properly defined')

        model = Sequential()
        model.add(Dense(layer_sizes[0], input_shape = (input_size,),name='hidden_layer_0'))

        for i, size in enumerate(layer_sizes):
            if i > 0 and i < len(layer_sizes) - 1:
                model.add(Dense(size, activation = 'relu', name='hidden_layer_{}'.format(i)))

        model.add(Dense(1, activation = 'linear',name='output')) 
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.alpha), metrics = ['mae'])
        return model

    def V(self,s):
        ''' Returns the value of being in state s TODO adjust'''

        if self.is_tabular:
            return self.value_function[s]
        else:
            binary_array = np.frombuffer(s) # convert from bytekode to np.array
            prediction_data = np.array([binary_array.reshape(binary_array.shape[0],)],dtype=np.float32) # reshape to match format required for forward pass
            
            try:
                estimate = self.value_function.model(prediction_data)
            except:
                raise Exception('Neural network in critic encountered a prediction error') 

            return estimate
            

    def calculateTD(self,s,s_next,r):
        ''' Temporal difference, reward + discount * V(s') - V(s) TODO adjust'''
        try: 
            return r + self.gamma * self.V(s_next) - self.V(s)

        except:
            if self.is_tabular:
                if s_next not in self.value_function:
                    self.value_function[s_next] = 0

                if s not in self.value_function:
                    self.value_function[s] = 0

            try:
                return r + self.gamma * self.V(s_next) - self.V(s)

            except:
                raise Exception('TD error cannot be calculated') 

    def resetEligibilities(self):
        self.e = dict()
        if not self.is_tabular:
            self.value_function.resetEligibilities()  

    def update(self,s,td_difference):
        '''TODO adjust'''
        if self.is_tabular:
            self.value_function[s] += self.alpha * td_difference * self.e[s]
            self.e[s] = self.gamma * self.decay * self.e[s]
        else:
            binary_array    = np.frombuffer(s) # convert from bytekode to np.array
            prediction_data = np.array([binary_array.reshape(binary_array.shape[0],)])
            feature         = K.constant( prediction_data ) # input feature: state
            target          = K.constant( np.array(td_difference + float(self.V(s))) ) # output target: this equals r + gamma * V(s')
            self.value_function.fit([feature],[target],td_difference,verbose=False) # calls on the CustomModel object - updates the weights in the neural network