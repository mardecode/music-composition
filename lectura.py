import pretty_midi as pm
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


def int2bin(n, maximo):
    # print(maximo, n)
    temp = [0] * int(maximo)
    temp[int(n - 1)] = 1
    return temp


def bin2int(n):
    return np.argmax(n)


duracionSet = [0.0625, 0.125, 0.25, 0.5,1, 2,4]

def aproximar(d):
    aprox = 0
    dist = abs(d - duracionSet[0])
    for i in range(1, len(duracionSet)):
        if abs(duracionSet[i] - d ) < dist:
            dist = abs(duracionSet[i] - d )
            aprox = i
    return aprox



def read_midis(path, instrument = 0):
    midis_p = os.listdir(path)
    # midis, tempos[0] = get_vector(path + "/" + midis_p[0], instrument)
    tempos = []
    dts = []
    ts = []
    ps = []

    for i in range(0, len(midis_p)):
        mid, T = get_vector(path + "/" + midis_p[i], instrument)
        for j in range(0, len(mid)):
            dts.append(mid[j][0])
            ts.append(mid[j][1])
            ps.append(mid[j][2])
        print(T)

    return (dts, ts, ps), tempos


def get_vector(file,n_instrument=0):
    midi = pm.PrettyMIDI(file)
    tempo = float(midi.estimate_tempo()/60)
    print('tempo ',midi.estimate_tempo()) ,
    print('instrumentos' , len(midi.instruments))

    nNotas = len(midi.instruments[n_instrument].notes)
    notas = np.zeros((nNotas,3),dtype='float64')

    notaAnterior = midi.instruments[n_instrument].notes[0]


    for i in range(0,nNotas):
        notaActual = midi.instruments[n_instrument].notes[i]
        dT = abs(notaActual.start - notaAnterior.start)
        T = abs(notaActual.end - notaActual.start )
        P = notaActual.pitch
        notas[i][0] = dT
        notas[i][1] = aproximar(T/tempo)
        notas[i][2] = P

        notaAnterior = notaActual

    return notas, tempo



(dts, ts, ps), tempos = read_midis("chopin",0)

ps = np.array(ps)
n_classes = np.max(ps.flatten())
ps = [int2bin(e, n_classes) for e in ps]

scaler_dt =MinMaxScaler(feature_range=(0,100))
scaler_t =MinMaxScaler(feature_range=(0,100))
# data = scaler.fit_transform(data)

dts = np.array(dts)
dts = dts.reshape(len(dts), 1)
dts = scaler_dt.fit_transform(dts)
dts = dts.reshape(len(dts), 1, 1)

ts = np.array(ts)
ts = ts.reshape(len(ts), 1)
ts = scaler_t.fit_transform(ts)
ts = ts.reshape(len(ts), 1, 1)

ps = np.array(ps)
ps = ps.reshape(len(ts), 1, 101)
tempos = np.array(tempos)


joblib.dump(dts,"datos/dts.pkl")
joblib.dump(ts,"datos/ts.pkl")
joblib.dump(ps,"datos/ps.pkl")
joblib.dump(tempos,"datos/tempos.pkl")


joblib.dump(scaler_dt,"datos/scaler_dt.pkl")
joblib.dump(scaler_t,"datos/scaler_t.pkl")
