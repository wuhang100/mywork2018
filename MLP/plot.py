# -*- coding: UTF-8 -*-
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential, load_model
import time
X_test = np.load('/home/wuhang/MLP/testinput.npy')
y_test = np.load('/home/wuhang/MLP/testout.npy')
model = models.load_model('/home/wuhang/MLP/model01.h5')
predicted1 = model.predict(X_test)
#print predicted1
model = models.load_model('/home/wuhang/MLP/model02.h5')
predicted2 = model.predict(X_test)
predicted = (predicted1 + predicted2)/2
num1 = 0
num2 = 30
for i in range(0,30):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test[num1:num2],label="Real")
    ax.legend(loc='upper left')
    plt.plot(predicted[num1:num2],label="Prediction")
    plt.legend(loc='upper left')
    plt.ylim(1200,1400)
    plt.xlabel('Time')
    plt.ylabel('T34')
    plt.grid(axis='y',linestyle='--',color='grey',linewidth='0.2')
    acc = 1-abs((predicted[num2]-y_test[num2])/y_test[num2])
    acc = 100*round(acc,5)
    plt.title('Current accuracy:'+str(acc)+'%')
    plt.savefig('/home/wuhang/html+js/picture/pyplot1.jpg')
#   plt.savefig('/home/wuhang/html+js/picture/pyplot2.jpg')
    plt.close('all')
    num1 = num1+1
    num2 = num2+1
    time.sleep(1)

