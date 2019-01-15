import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import datetime
X_train = np.load('/home/wuhang/MLP/inputdata11.npy')
y_train = np.load('/home/wuhang/MLP/outdata11.npy')
model = Sequential()
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adam',metrics=['mae', 'mape'])
starttime = datetime.datetime.now()
model.fit(
    X_train, y_train,
    batch_size=50, epochs=150,validation_split=0.1, verbose=0)
endtime = datetime.datetime.now()
model.save('/home/wuhang/MLP/model02.h5')
print 'training complete, the time is: '+str((endtime - starttime).seconds)


