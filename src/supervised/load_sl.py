from sl import SupervisedTau

obj = SupervisedTau()
obj.loadData('dataset_train_1111.npy')
print(obj.df)

print(obj.getNormalizedData())