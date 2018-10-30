import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import numpy as np
import pandas as pd

batch_size = 1
nb_inputs = 25
nb_classes = 5
epochs = 300

train = pd.read_csv('shapes_train.dat', sep=" ", header=None)

x_train = (train.ix[:,:train.shape[1]-nb_classes-1].values).astype('float32')
labels = (train.ix[:,nb_inputs:train.shape[1]-1].values).astype('int32')

model = Sequential()
model.add(Dense(nb_inputs, activation='relu', input_shape=(nb_inputs,)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, clipnorm=1.),
              metrics=['accuracy'])

history = model.fit(x_train, labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

test = pd.read_csv('shapes_test.dat', sep=" ", header=None)
x_test = (test.ix[:,:test.shape[1]-1].values).astype('float32')
outputs = model.predict(x_test)

print(outputs)

prediction = {}
prediction['confidence'] = float(max(outputs[0]))
df = pd.DataFrame(data=outputs)
prediction['class'] = df.values[0].tolist().index(prediction['confidence'])

print(prediction)