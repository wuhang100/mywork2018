import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import datetime
X_train = np.load('/home/wuhang/MLP/inputdata01.npy')
y_train = np.load('/home/wuhang/MLP/outdata01.npy')
model = Sequential()
model.add(Dense(180,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(180,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adam',metrics=['mae', 'mape'])
starttime = datetime.datetime.now()
model.fit(
    X_train, y_train,
    batch_size=30, epochs=120,validation_split=0.15, verbose=0)
endtime = datetime.datetime.now()
model.save('/home/wuhang/MLP/model01.h5')
print 'training complete, the time is '+str((endtime - starttime).seconds)+'s'


