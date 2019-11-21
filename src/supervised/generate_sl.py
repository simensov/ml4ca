from sl import SupervisedTau

obj = SupervisedTau()
ad = 21
ud = 21
obj.generateData(ad,ud)
obj.saveData('dataset_train_{}{}.npy'.format(ad,ud))