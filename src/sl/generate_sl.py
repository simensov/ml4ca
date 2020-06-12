#!/usr/bin/env python3
'''
Generates a dataset with self chosen discretization
@author: Simen Oevereng, simensem@gmail.com, December 2019
'''

from SupervisedTau import SupervisedTau

for d in [31]: #,25,31]: # MUST BE ODD NUMBER!
    obj = SupervisedTau()
    ad = d # Azimuth angle discretiation
    ud = d # Thrust input discretization
    obj.generateData(ad,ud)
    obj.saveData('dataset_train_{}{}.npy'.format(ad,ud)) # Save as .npy file