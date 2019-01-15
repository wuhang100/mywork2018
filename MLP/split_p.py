import numpy as np
import mysql.connector
from sklearn import preprocessing
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
query = ("SELECT P34 FROM test0.2016stablepoint LIMIT 1,25000")
cursor.execute(query)
result=cursor.fetchall()
y_list = np.array(result)
#print y_list
#print np.hstack((train_x,y_list))
x_list = np.hstack((train_x,y_list))
#result = []
#sequence_length = 5
#for index in range(len(x_list) - sequence_length + 1):
#    result.append(x_list[index: index + sequence_length])
#result = np.array(result)
#np.random.seed(10)
#np.random.shuffle(result)
#print result
row1 = int(round(0.8 * x_list.shape[0]))
row2 = int(round(0.8 * x_list.shape[0]))
row3 = int(round(0.9 * x_list.shape[0]))
train = x_list[:row1, :]
test = x_list[row2:row3, :]
np.random.seed(10)
np.random.shuffle(train)
X_train = train[:,0:6]
y_train = train[:,-1]
np.random.seed(10)
np.random.shuffle(test)
X_test = test[:,0:6]
y_test = test[:,-1]
y_train = np.reshape(y_train, (y_train.shape[0],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
scaler = preprocessing.StandardScaler().fit(x_list)
train_x = scaler.transform(x_list)
#print X_train
#print y_train
print X_test[1,:]
np.save('/home/wuhang/MLP/testinput_P.npy',X_test)
np.save('/home/wuhang/MLP/testout_P.npy',y_test)
