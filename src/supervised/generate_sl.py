from SupervisedTau import SupervisedTau

for d in [5,11,21,25,31]:
    obj = SupervisedTau()
    ad = d
    ud = d
    obj.generateData(ad,ud)
    obj.saveData('dataset_train_{}{}.npy'.format(ad,ud))