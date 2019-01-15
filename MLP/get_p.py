from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential, load_model
import time
X_test = np.load('/home/wuhang/MLP/testinput_P.npy')
y_test = np.load('/home/wuhang/MLP/testout_P.npy')
model = models.load_model('/home/wuhang/MLP/model_p.h5')
#print X_test[1,:]
predicted = model.predict(X_test)
#print predicted1
num1 = 0
num2 = 30
x_list = []
a = 0
for i in range(0,50):
    a = a+1
    datastr = '02-21 11:'+str(a)
    x_list.append(datastr)
#print x_list[0:30]
import matplotlib.dates as md
import dateutil

for i in range(0,19):
    datestrings = x_list[num1:num2]
    dates = [dateutil.parser.parse(s) for s in datestrings]

    plt.figure(figsize=(10,5))
    plt_data = y_test[num1:num2]
    plt_data2 = predicted[num1:num2]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=90 )

    ax=plt.gca()
    ax.set_xticks(dates)

    xfmt = md.DateFormatter('%m-%d %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(dates,plt_data, label="Real")
    plt.plot(dates,plt_data2, label="Prediction")
    plt.legend(loc='upper left')
    plt.xticks(fontsize=8)
    plt.ylim(42,50)
    plt.ylabel('P34')
    plt.grid(axis='y',linestyle='--',color='grey',linewidth='0.2')
    acc = 1-abs((predicted[num2]-y_test[num2])/y_test[num2])
    acc = 100*round(acc,5)
    plt.title('Input:P1,T1,P2,T2,n1,n2, current accuracy:'+str(acc)+'%',loc='left')
#   plt.savefig('/home/wuhang/html+js/picture/pyplot1.jpg')
    plt.savefig('/home/wuhang/html+js/picture/pyplot2.jpg',dpi=500)
    plt.close('all')
    num1 = num1+1
    num2 = num2+1
    time.sleep(3)
