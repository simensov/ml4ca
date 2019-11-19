
# import importlib

# moduleName = 'sl'
# importlib.import_module(moduleName)

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils import model_to_dot

from sl import SupervisedTau

st = SupervisedTau()

st.loadData('dataset_train.npy')
# st.generateData()

# st.displayData()

dataset = st.data


# Training data and test data
X = dataset[:,0:3]
Y = dataset[:,3:]

print(X)
print(Y)

def nn_model(num_hidden=3):
    # create model
    # alternatives to test: more Dense layers (deeper), more hidden neurons (wider)
    model = Sequential()

    # Hidden layers
    model.add(Dense(num_hidden, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_hidden,kernel_initializer='normal', activation='relu'))

    # Output layer
    model.add(Dense(6, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adadelta')

    plot_model(model, to_file='model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    return model

# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=nn_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# TODO add test data