#!/usr/bin/env python3

from SupervisedTau import SupervisedTau

for d in [51]: #,25,31]:
    obj = SupervisedTau()
    ad = d
    ud = d
    obj.generateData(ad,ud)
    obj.saveData('dataset_train_{}{}.npy'.format(ad,ud))