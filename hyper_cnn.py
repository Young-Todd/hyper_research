import keras
from keras import backend as K

from hyperopt import fmin, tpe, hp, STATUS_OK Trials
from hyperopt.mongoexp import MongoTrials
from sklearn.metrics import roc_auc_score
from pprint import pprint
import sys

def build_search_space():
    space = {
        'epochs' :  100,
        'batch_size' : hp.uniform('batch_size', 75, 125),
        'filters': hp.choice('filters', [64, 128]),
        'kernel_size': hp.choice('kernel_size', [(2,2), (3,3)]),
        #'kernel_size': hp.randint('kernel_size', 4),
        'dropout1': hp.uniform('dropout1', .25,.75),
        'dropout2': hp.uniform('dropout2',  .25,.75),
        'activation': hp.choice('activation', ['relu']),
        'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
    }
    return space

def run_trials(model, space, trials_step, max_trials):
    """
    Allows us to load previous trials of the experiment to continue the search and gives the abilitiy to 
    stop the search manually without the fear of losing information.
    """
    
    # The following two lines are for debugging.
    trials_step = trials_step  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = max_trials  # initial max_trials. put something small to not have to wait
    
    
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("my_model.hyperopt", "rb"))x
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()
    
    best = fmin(fn=mymodel, space=model_space, algo=tpe.suggest, max_evals=max_trials, trials=trials)

    print("Best:", best)
    
    # save the trials object
    with open(_model + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)

    # loop indefinitely and stop whenever you like
    while True:
        run_trials()


def data():
    '''
        Data providing function:
        
        Make sure to have every relevant import statement included here and return data as
        used in model function below. This function is separated from model() so that hyperopt
        won't reload data for each evaluation run.
    '''
    from keras.datasets import mnist
    from keras.utils import np_utils
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    data_list = [input_shape, num_classes, x_train, y_train, x_test, y_test]
    return data_list

def cnn(space, input_shape, num_classes, x_train, y_train, x_test, y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Sequential
    from keras.models import Model
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop
    
    print('Searching over space:\n')
    pprint.pprint(space)
    model = Sequential()
    model.add(Conv2D(28,
                     kernel_size=space['kernel_size'],
                     activation=space['activation'],
                     input_shape=input_shape))
    model.add(Conv2D(filters=space['filters'],
                     kernel_size=space['kernel_size'],
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=space['activation']))
    model.add(Dropout(space['dropout1']))
    model.add(Dense(num_classes, activation='softmax'))
                     
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
                     
    model.fit(x_train, y_train,
              batch_size=space['batch_size'],
              epochs=space['epochs'],
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
                     
    return {'loss': -score[1], 'status': STATUS_OK}

if __name__ == "__main__":
    space = build_search_space()
    data_list = data()
    
    # Use Trials() for a single machine
    #trials = Trials()
    # Use MongoTrials for the distributed setting.
    trials = MongoTrials('mongo://localhost:1234/experiment_db/jobs', exp_key='exp1')
    best = fmin(cnn(space, *data_list), space, algo=tpe.suggest, max_evals=50, trials=trials)
    print 'best: '
    print best
