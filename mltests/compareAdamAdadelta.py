# Regression Example With Boston Dataset: Standardized and Wider
# From https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/, 9. nov 2019
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset - put on path or in current working dir
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# evaluate model with standardized dataset
print('Testing adam')
estimators1 = []
estimators1.append(('standardize', StandardScaler()))
estimators1.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=1)))
pipeline1 = Pipeline(estimators1)
kfold1 = KFold(n_splits=10)
results1 = cross_val_score(pipeline1, X, Y, cv=kfold1)

# test optimizer adadelta
def wider_model_2():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adadelta')
	return model

# evaluate model with standardized dataset
print('Testing adadelta')
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model_2, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)

# Compare
print("Wider: %.2f (%.2f) MSE" % (results1.mean(), results1.std()))
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))    