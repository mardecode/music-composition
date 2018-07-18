from keras.models import load_model
from sklearn.externals import joblib
from funciones import *


dt = joblib.load("datos/dts.pkl")
t = joblib.load("datos/ts.pkl")
p = joblib.load("datos/ps.pkl")
scaler_t = joblib.load("datos/scaler_t.pkl")
scaler_dt = joblib.load("datos/scaler_dt.pkl")


modelo = load_model("modelos/modelo_experimental.h5")


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





#======================= prediciones====================

tam = 1000

o1, o2, o3 = modelo.predict([i_dt[:tam], i_t[:tam], i_p[:tam]])

o1 = scaler_dt.inverse_transform(o1)
o2 = scaler_t.inverse_transform(o2)

midi = []

for i in range(0, tam):
    midi.append([o1[i][0], o2[i][0], bin2int(o3[i])])

midi = np.array(midi)

midi_out = reconstruirCancion(midi, 3)

midi_out.write("confe.mid")
