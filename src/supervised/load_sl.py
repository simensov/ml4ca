from sl import SupervisedTau

obj = SupervisedTau()
obj.loadData('dataset_train_55.npy')
print(obj.df)