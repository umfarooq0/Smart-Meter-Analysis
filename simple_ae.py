# https://machinelearningmastery.com/autoencoder-for-classification/
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
# define datase
  
'''
dir = '/home/usman/Documents/Smart-Meter-analysis/archive/hhblock_dataset/hhblock_dataset'
list = os.listdir(dir) # dir is your directory path
number_files = len(list)

## list of filenames
_, _, filenames = next(os.walk(dir))
'''
df_ = pd.read_csv('/home/usman/Documents/Smart-Meter-analysis/total_hhblock.csv')
'''
df_ = pd.DataFrame()
i = 0
for x in filenames:
    i += 1
    print()
    uf = pd.read_csv(dir + '/' + x)
    df_ = pd.concat([df_,uf])
    print(i)
'''


'''
# check for stationarity 
results = [adfuller(df_[x]) for x in range(df.shape[0])]

def test_ts_stat(x):
    #x should be the output statistic from ADF test
    one = x[4]['1%']
    five = x[4]['5%']
    ten = x[4]['10%']

    if x[0] < one or x[0] < five or x[0] < ten:
        return True
    else:
        return False

results_check = [test_ts_stat(x) for x in results]

result = adfuller(df_[0])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

'''
bottleneck_size = 5
epochs = 40
batch_size = 8

ids_ = df_.LCLid.unique()

i = 0

def ae(df_,id_no,bottleneck_size,epochs,batch_size):
    '''
    df_ : Complete dataset
    id_no: LCLid that you want to train
    bottleneck_size,
    epochs,
    batch_size
    
    '''
    ver_ = list(df_.LCLid.unique())
    t = MinMaxScaler()
    X = df_[df_.LCLid == id_no ]
    X = X.iloc[:,3:]
    X_train = t.fit_transform(X)

    #check nan in X_train
    where_ = [np.isnan(X_train[x]) for x in range(len(X_train))]

    where_nan = np.where(where_)[0]

    if where_nan.size > 0:
        X_train = np.delete(X_train, where_nan[0], 0)
    
    n_inputs = X_train.shape[1]

    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    bottleneck = Dense(bottleneck_size)(e)
    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    # plot the autoencoder
    #plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
    print(ver_.index(id_no))

    return model.get_weights()[13]

import time

start = time.time()
run_ae_one = [ae(df_,x,5,70,12) for x in ids_[:1000]]
end = time.time() - start

features_ = pd.DataFrame(run_ae)
features_.to_csv('features_.csv')