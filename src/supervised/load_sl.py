#!/usr/bin/env python3

from SupervisedTau import SupervisedTau
import numpy as np
np.set_printoptions(precision=3) # print floats as decimals with 3 zeros

import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import pandas as pd

pd.set_option('precision', 3)


'''
Load data and inspect 
'''

st = SupervisedTau()
st.loadData('dataset_train_1111.npy')
dataset = st.data

input_size = 9
label_size = 6
X = dataset[:,0:input_size]
Y = dataset[:,input_size:]

num_datapoints = X.shape[0]
num_datapoints = 400

### Plot the looks of tau
plt.figure()
plt.subplot(321)
taux = X[:,6:7]
plt.hist(taux,bins=num_datapoints)
plt.xlabel('Force in surge [N]')
plt.subplot(323)
tauy = X[:,7:8]
plt.hist(tauy,bins=num_datapoints)
plt.xlabel('Force in sway [N]')
plt.subplot(325)
taup = X[:,8:]
plt.hist(taup,bins = num_datapoints)
plt.xlabel('Moment in yaw [Nm]')


####### TEST STANDARDIZING TO ZERO MEAN, STD == 1
taus = [taux, tauy, taup]
scaled_data = []

def standardize(data):
    '''
    Return a standardized array of data (Mx1 array) 
    '''

    return (data - np.mean(data)) / np.std(data)

for force in taus:
    scaled_data.append(standardize(force))


plt.subplot(322)
taux_std = scaled_data[0]
plt.hist(taux_std,bins=400)
plt.xlabel('Force in surge, standardized')
plt.subplot(324)
tauy_std = scaled_data[1]
plt.hist(tauy_std,bins=400)
plt.xlabel('Force in sway, standardized')
plt.subplot(326)
taup_std = scaled_data[2]
plt.hist(taup_std,bins = 400)
plt.xlabel('Moment in yaw, standardized')

#### TEST THE ROBUST SCALER (looks similar to standardization)
scaler = RobustScaler() 
taux_rob = scaler.fit_transform(taux)
tauy_rob = scaler.fit_transform(tauy)
taup_rob = scaler.fit_transform(taup)

plt.figure()
plt.subplot(321)
plt.hist(taux_rob,bins=400)
plt.subplot(323)
plt.hist(tauy_rob,bins=400)
plt.subplot(325)
plt.hist(taup_rob,bins=400)

### TEST NORMALIZATION

def normalize(data):
    '''
    Return the normalized version of a column vector
    '''
    return (data - np.min(data)/ (np.max(data) - np.min(data)))

norm_data = []
for force in taus:
    norm_data.append(normalize(force))

plt.subplot(322)
taux_norm = norm_data[0]
plt.hist(taux_norm,bins=400)
plt.xlabel('Force in surge, normalized')
plt.subplot(324)
tauy_norm = norm_data[1]
plt.hist(tauy_norm,bins=400)
plt.xlabel('Force in sway, normalize')
plt.subplot(326)
taup_norm = norm_data[2]
plt.hist(taup_norm,bins = 400)
plt.xlabel('Moment in yaw, normalize')

# TODO augment the data such that the lesser repeated values appear more often

plt.show()


