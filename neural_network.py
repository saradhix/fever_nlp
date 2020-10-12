import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
name="Neural Network(Simple Layer)"
model = Sequential()
def fit_predict(X_train, y_train, X_test, y_test,  cr=True, cm=False):
    OPTIMIZER = 'rmsprop'
    DP=0.2
    EPOCHS=10
    VERBOSE=1
    #print("fit_predict:X-train", len(X_train), len(X_train[0]))
    #print(X_train[0])
    num_classes = len(set(y_train))
    if num_classes == 2:
        binary=True
        loss = 'binary_crossentropy'
    else:
        binary=False
        loss = 'categorical_crossentropy'

    model.add(Dense(600, kernel_initializer='uniform', activation='relu'))
    if not binary:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    model.compile(loss=loss, optimizer=OPTIMIZER, metrics=['accuracy'])

    if not binary:
        y_train = [label_to_vector(y, num_classes) for y in y_train]
    #y_test_v = [label_to_vector(y, num_classes) for y in y_test]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    #model.summary()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=1024, verbose=VERBOSE)
    y_pred = model.predict(X_test)
    if not binary:
        y_pred = [vector_to_label(vector) for vector in y_pred]
    else:
        y_pred = [ int(round(i[0])) for i in y_pred]

    if cm: print( confusion_matrix(y_test, y_pred))
    if cr: print( classification_report(y_test, y_pred, digits=4))
    #Dump the model
    #pickle_file = 'neural_network_model.pickle'
    #pickle.dump(model, open(pickle_file,"wb"))
    return y_pred

def label_to_vector(y, num_classes):
    vector=[0 for i in range(num_classes)]
    vector[int(y)]=1
    return vector

def vector_to_label(vector):
    vector = vector.tolist()
    return vector.index(max(vector))

def predict(X_test):
    y_pred = model.predict(X_test)
    y_pred = [vector_to_label(vector) for vector in y_pred]
    return y_pred
