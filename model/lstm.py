import os
import numpy as np
from keras.layers import Dense, LSTM
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.externals import joblib
from datetime import datetime

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

    num_states = len(set(y_train_condensed[:].flatten()))
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
        x_temp = x_train[i:i+max_len]
        end = is_end[i:i+max_len]
        y_temp = y_train[i+max_len]
        if end.max() < 1:
            x.append(x_temp)  # max_len X 666
            y.append(y_temp)  # 1 X 4

    training_x = np.array(x)  # num_samples X max_len X 666
    training_y = np.array(y)  # num_samples X 4

    x2 = list()
    y2 = list()
    for i in range(0, num_test_samples-max_len, step):
        x_temp = x_test[i:i+max_len]
        end = is_end[i:i+max_len]
        y_temp = y_test[i+max_len]
        if end.max() < 1:
            x2.append(x_temp)  # max_len X 666
            y2.append(y_temp)  # 1 X 4
            
    testing_x = np.array(x2)  # num_samples X max_len X 666
    testing_y = np.array(y2)  # num_samples X 4

    return training_x, training_y, testing_x, testing_y
    

def lstm_algo(trainingdataset='', traininglabels='', testingdataset='', testinglabels='', lengths='', logger=''):

    x, y, x2, y2 = get_data(x_train=trainingdataset, y_train_condensed=traininglabels, x_test=testingdataset,
                            y_test_condensed=testinglabels, lengths=lengths)

    logs_run = 'runs'
    if not os.path.exists(logs_run):
        os.mkdir(logs_run)

    model = get_model()
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    score = [0, 0]

    # for iteration in range(1, 10):

    filename = 'kerasmodel_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.h5'
    filepath = os.path.join(logs_run, filename)

    # msg = 'starting iteration: {0}'.format(iteration)
    # logger.getLogger('tab.regular').info(msg)
    model.fit(x, y, batch_size=128, nb_epoch=1)

    logger.getLogger('tab.regular').info('running evaluate')
    loss_and_metrics = model.evaluate(x2, y2)

    msg = 'error and accuracy: {0}'.format(loss_and_metrics)
    logger.getLogger('tab.regular').info(msg)

    if loss_and_metrics[0] > score[0]:
        score = loss_and_metrics
        model.save(filepath)

