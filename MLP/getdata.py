from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential, load_model
X_test = np.load('/home/wuhang/MLP/testinput.npy')
y_test = np.load('/home/wuhang/MLP/testout.npy')
model = models.load_model('/home/wuhang/MLP/model01.h5')
predicted1 = model.predict(X_test)
#print predicted1
model = models.load_model('/home/wuhang/MLP/model02.h5')
predicted2 = model.predict(X_test)
predicted = (predicted1 + predicted2)/2
acc = np.ones((y_test.shape[0],1))-abs((predicted-y_test)/y_test)
acc = 100*round(np.mean(acc),5)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test[0:50],label="Real")
ax.legend(loc='upper left')
plt.plot(predicted[0:50],label="Prediction",linestyle='--')
plt.legend(loc='upper left')
plt.ylim(1200,1400)
plt.xlabel('data point')
plt.ylabel('temperature (K)')
plt.grid(axis='y',linestyle='--',color='grey',linewidth='0.2')
plt.title('Accuracy:'+str(acc)+'%')
plt.show()

