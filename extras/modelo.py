from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.layers import LSTM, CuDNNLSTM
from keras.optimizers import RMSprop
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = joblib.load("notas.pkl")

print(data.shape)

n = len(data)

data_x = data[0 : n - 1]
data_y = data[1 : n]

# scaler =MinMaxScaler(feature_range=(0,1))
# data_x = scaler.fit_transform(data_x)

data_x = data_x.reshape(n - 1, 1, 3)


print(data_x.shape, " - ",data_y.shape)

# sgd = SGD(lr=0.1)


model = Sequential()

model.add(CuDNNLSTM(64, input_shape=(1, 3)))
model.add(Dense(32, activation= "relu"))
model.add(Dense(3, activation= "relu"))
# model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer="SGD", metrics=['mae', 'acc'])
model.save("modelo.h5")



model.fit(data_x, data_y, batch_size=100, epochs=10)
# score = model.evaluate(x_test, y_test, batch_size=16)
