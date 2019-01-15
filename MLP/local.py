import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential, load_model
import mysql.connector
from keras.optimizers import Adam
from sklearn import preprocessing
import datetime
cnx = mysql.connector.connect(user='root', password='941012',
                              host='127.0.0.1',
                              database='test0')
cursor = cnx.cursor()
query = ("SELECT n1,n2,P1,T1,P2,T2 FROM test0.2016stablepoint LIMIT 1,25000")
cursor.execute(query)
result=cursor.fetchall()
x_list = np.array(result)
scaler = preprocessing.StandardScaler().fit(x_list)
train_x = scaler.transform(x_list)
#print train_x
query = ("SELECT T34 FROM test0.2016stablepoint LIMIT 1,25000")
cursor.execute(query)
result=cursor.fetchall()
y_list = np.array(result)
#y_list = y_list-1300*np.ones((y_list.shape[0], 1))
#print y_list
#print np.hstack((train_x,y_list))
x_list = np.hstack((train_x,y_list))
row1 = int(round(0.2 * x_list.shape[0]))
row2 = int(round(0.2 * x_list.shape[0]))
row3 = int(round(0.25 * x_list.shape[0]))
train = x_list[:row1, :]
test = x_list[row2:row3, :]
np.random.seed(10)
np.random.shuffle(train)
X_train = train[:,0:6]
y_train = train[:,-1]
np.random.seed(1)
np.random.shuffle(test)
X_test = test[:,0:6]
y_test = test[:,-1]
y_train = np.reshape(y_train, (y_train.shape[0],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
cursor.close()
cnx.close()
#print X_train
#print y_train
model = Sequential()
model.add(Dense(180,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(180,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse', optimizer='adam',metrics=['mae', 'mape'])
starttime = datetime.datetime.now()
train_history = model.fit(
                          X_train, y_train,
                          batch_size=30, epochs=120,validation_split=0.15, verbose=0)
endtime = datetime.datetime.now()
#def show_train_history(train_history,train,validation):
#    plt.plot(train_history.history[train])
#    plt.plot(train_history.history[validation])
#    plt.title('Train History')
#    plt.ylabel(train)
#    plt.xlabel('Epoch')
#    plt.legend(['train','validation'],loc='upper left')
#    plt.ylim(0,500)
#    plt.show()
#show_train_history(train_history,'loss','val_loss')
print 'TIME: '+str((endtime - starttime).seconds)+'s'
predicted = model.predict(X_test)
predicted_pro = (np.mean(y_test)-np.mean(predicted))*np.ones(50)+predicted[0:50,0]
#predicted = predicted+1300*np.ones((predicted.shape[0], 1))
#y_test = y_test+1300*np.ones((y_test.shape[0], 1))
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
a = np.load('/home/wuhang/LSTM_code/testout.npy')
#ax.plot(a[0:50],label="Real")
ax.plot(y_test[0:50],label="Real")
ax.legend(loc='upper left')
plt.plot(predicted[0:50],label="Prediction",linestyle='--')
plt.legend(loc='upper left')
plt.ylim(1200,1400)
plt.xlabel('data point')
plt.ylabel('temperature (K)')
acc = np.ones((y_test.shape[0],1))-abs((predicted-y_test)/y_test)
acc = round(np.mean(acc),5)
plt.title('Accuracy:'+str(100*acc)+'%')
plt.grid(axis='y',linestyle='--',color='grey',linewidth='0.2')
plt.show()
from keras.utils import plot_model
plot_model(model, to_file='/home/wuhang/thesispaper/modelMLP.png', show_shapes=True, show_layer_names=True)
#model.save('/home/wuhang/thesispaper/modelMLP01.h5')
