from keras.layers import Input, Dense, LSTM, concatenate, CuDNNLSTM
from keras.models import Model,load_model
from sklearn.externals import joblib
import numpy as np

dt = joblib.load("datos/dts.pkl")
t = joblib.load("datos/ts.pkl")
p = joblib.load("datos/ps.pkl")


n = len(dt)

i_dt = dt[0 : n - 1]
o_dt = dt[1 : n]

i_t = t[0 : n - 1]
o_t = t[1 : n]

i_p = p[0 : n - 1]
o_p = p[1 : n]

n = n - 1

o_dt = o_dt.reshape(n , 1)
o_t = o_t.reshape(n, 1)
o_p = o_p.reshape(n, 101)

n_classes = 101


# i_dt = np.ones(shape = (n, 1, 1))
# o_dt = np.ones(shape = (n, 1))
#
# i_t = np.ones(shape = (n, 1, 1))
# o_t = np.ones(shape = (n, 1))
#
# i_p = np.ones(shape = (n, 1, n_classes))
# o_p = np.ones(shape = (n, n_classes))


input_dt = Input(shape = (1, 1), name = "input_dt")
lstm_dt = CuDNNLSTM(32, return_sequences = True)(input_dt)

input_t = Input(shape = (1, 1), name = "input_t")
lstm_t = CuDNNLSTM(64, return_sequences = True)(input_t)

input_p = Input(shape = (1, n_classes), name = "input_p")
lstm_p = CuDNNLSTM(100, return_sequences = True)(input_p)


hold_input = concatenate([lstm_dt, lstm_t, lstm_p])


main_lstm = CuDNNLSTM(256, return_sequences = True)(hold_input)

lstm_dt_o = CuDNNLSTM(32)(main_lstm)
dt_output = Dense(1, activation = "relu", name = "dt_output")(lstm_dt_o)

lstm_t_o = CuDNNLSTM(64)(main_lstm)
t_output = Dense(1, activation = "relu", name = "t_output")(lstm_t_o)

lstm_p_o = CuDNNLSTM(100)(main_lstm)
p_output = Dense(n_classes, activation = "sigmoid", name = "p_output")(lstm_p_o)


model = Model(inputs=[input_dt, input_t, input_p], outputs=[dt_output, t_output, p_output])

model.compile(optimizer='rmsprop',
              loss={'dt_output': 'mean_squared_error', 't_output': 'mean_squared_error', 'p_output': 'binary_crossentropy'})

model = load_model("modelos/modelo_experimental6.h5")
model.fit({'input_dt': i_dt, 'input_t': i_t, 'input_p': i_p},
          {'dt_output': o_dt, 't_output': o_t, 'p_output': o_p},
          epochs=10, batch_size=130)

model.summary()

model.save("modelos/modelo_experimental7.h5")
# metrics = {'dt_output': ['mse', 'accuracy'], 't_output': ['mse', 'accuracy'], 'p_output': ['mse', 'accuracy']},
a, b, c = model.predict([i_dt[:3], i_t[:3], i_p[:3]])
print("p: ", c)
print("dt: ",a)
print("t: ",b)
