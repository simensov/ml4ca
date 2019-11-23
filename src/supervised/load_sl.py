from SupervisedTau import SupervisedTau
import numpy as np
np.set_printoptions(precision=3) # print floats as decimals with 3 zeros

import pandas as pd

pd.set_option('precision', 3)

obj = SupervisedTau()
obj.loadData('dataset_train_2121.npy')
print(obj.df)