import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn import preprocessing
import datetime
X_train4D = np.load('/home/wuhang/cnn/inputdata01.npy')
y_train = np.load('/home/wuhang/cnn/outdata01.npy')
model = Sequential()
model.add(Conv2D(filters=4,
                kernel_size=(2,2),
                padding='same',
                input_shape=(6,6,1),
                activation='relu'))

model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adam',metrics=['mae', 'mape'])
start = datetime.datetime.now()
model.fit(
    X_train4D, y_train,
    batch_size=50, epochs=110,validation_split=0.1, verbose=0)
end = datetime.datetime.now()
model.save('/home/wuhang/cnn/model01.h5')
print 'training complete, the time is '+str(end-start)


