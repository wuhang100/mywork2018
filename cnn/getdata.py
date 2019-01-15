from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.models import Sequential, load_model
X_test = np.load('/home/wuhang/cnn/testinput.npy')
y_test = np.load('/home/wuhang/cnn/testout.npy')
model = models.load_model('/home/wuhang/cnn/model01.h5')
predicted1 = model.predict(X_test)
from keras.utils import plot_model
plot_model(model, to_file='/home/wuhang/cnn/model.png', show_shapes=True, show_layer_names=True)
#print predicted1
model = models.load_model('/home/wuhang/cnn/model02.h5')
predicted2 = model.predict(X_test)
predicted = (predicted1 + predicted2)/2
acc = np.ones((y_test.shape[0],1))-abs((predicted-y_test)/y_test)
acc = 100*round(np.mean(acc),5)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test[0:50],label="Real")
ax.legend(loc='upper left')
plt.plot(predicted[0:50],label="Prediction")
plt.legend(loc='upper left')
plt.ylim(1100,1450)
plt.title('Accuracy:'+str(acc)+'%')
plt.show()

