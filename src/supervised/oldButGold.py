

### TESTTESTTEST - this was used to test performance with example found online. Not necessarily compatible with new keras version
if False:
    # load dataset - put on path or in current working dir
    dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]
    # np.random.shuffle(dataset)
    # scaler = StandardScaler()
    # stdsc = scaler.fit(dataset)
    # dataset_scaled = scaler.transform(dataset)
    # X = dataset_scaled[:,0:13]
    # Y = dataset_scaled[:,13]


    print('Traning model on {} datapoints'.format(X.shape[0]))
    nn = FFNeuralNetwork(13,1,20,1)

    def test_model():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model
    
    #nn.history = nn.model.fit(X,Y,validation_split = 0.1, epochs = 100, batch_size = 5, verbose = 0, shuffle = True)
    #nn.plotHistory()

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=nn.nn_model, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print(results.mean(), results.std())
### TESTTESTTEST