# coding: utf-8
import h5py
import numpy as np
import keras
from keras.layers import Dense, LSTM
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

MAX_LEN = 500


def get_model():
    model = Sequential()
    
    model.add(LSTM(output_dim=32, input_shape=(MAX_LEN, 665)))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=30))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=4))
    model.add(Activation("softmax"))

    return model 


def get_data(x_train, y_train_condensed, x_test, y_test_condensed, lengths):
    
    # num_samples x 1
    # but we need num_samples x 3

    num_states = 4
    num_train_samples = y_train_condensed.shape[0]
    num_test_samples = y_test_condensed.shape[0]

    y_train = np.zeros((num_train_samples, num_states))
    y_test = np.zeros((num_test_samples, num_states))

    y_train[np.arange(num_train_samples), y_train_condensed[:].astype(np.int).reshape(num_train_samples)] = 1
    y_test[np.arange(num_test_samples), y_test_condensed[:].astype(np.int).reshape(num_test_samples)] = 1

    cum_sum_lengths = np.cumsum(lengths)
    is_end = np.zeros((num_train_samples))
    is_end[cum_sum_lengths[:-1]] = 1
    
    step = 300
    max_len = MAX_LEN

    x = list()
    y = list()
    for i in range(0, num_train_samples-max_len, step):
        print i
        x = x_train[i:i+max_len]
        end = is_end[i:i+max_len]
        y = y_train[i+max_len]
        if end.max() < 1:
            x.append(x)  # max_len X 666
            y.append(y)  # 1 X 4
            
    x = np.array(x)  # num_samples X max_len X 666
    y = np.array(y)  # num_samples X 4

    x2 = list()
    y2 = list()
    for i in range(0, num_test_samples-max_len, step):
        print i
        x = x_test[i:i+max_len]
        end = is_end[i:i+max_len]
        y = y_test[i+max_len]
        if end.max() < 1:
            x2.append(x)  # max_len X 666
            y2.append(y)  # 1 X 4
            
    x2 = np.array(x2)  # num_samples X max_len X 666
    y2 = np.array(y2)  # num_samples X 4
    return x, y, x2, y2
    

def lstm_algo(trainingdataset='', traininglabels='', testingdataset='', testinglabels='', lengths=''):

    x, y, x2, y2 = get_data(x_train=trainingdataset, y_train_condensed=traininglabels, x_test=testingdataset,
                            y_test_condensed=testinglabels, lengths=lengths)

    model = get_model()
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    for iteration in range(1, 10):
        print "starting iteration: " + str(iteration)
        model.fit(x, y, batch_size=128, nb_epoch=1)
        
        print "running evaluate"
        print model.evaluate(x2, y2)
