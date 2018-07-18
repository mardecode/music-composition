import pretty_midi as pm
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
from sklearn.externals import joblib


def int2bin(n, maximo):
    # print(maximo, n)
    temp = [0] * int(maximo)
    temp[int(n - 1)] = 1
    return temp


def bin2int(n):
    return np.argmax(n)

def plot_piano_roll(midi, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(midi.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(start_pitch))

duracionSet = [0.0625, 0.125, 0.25, 0.5,1, 2,4]




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
        tempos.append(T)

    return (dts, ts, ps), tempos



def aproximar(d):
    aprox = 0
    dist = abs(d - duracionSet[0])
    for i in range(1, len(duracionSet)):
        if abs(duracionSet[i] - d ) < dist:
            dist = abs(duracionSet[i] - d )
            aprox = i
    return aprox

def aproximarInversa(i, tempo):
    i = np.clip(i , 0, 4)
    return duracionSet[int(i)]*tempo

def read_midis(path, instrument = 0):
    midis_p = os.listdir(path)
    tempos = [0.0]
    midis, tempos[0] = get_vector(path + "/" + midis_p[0], instrument)


    for i in range(1, len(midis_p)):
        mid, T = get_vector(path + "/" + midis_p[i], instrument)
        midis = np.concatenate([midis, mid])
        tempos.append(T)

    return midis, tempos


def get_vector(file,n_instrument=0):
    midi = pm.PrettyMIDI(file)
    tempo = float(midi.estimate_tempo()/60)
    print('tempo ',midi.estimate_tempo()) ,
    print('instrumentos' , len(midi.instruments))

    nNotas = len(midi.instruments[n_instrument].notes)
    notas = np.zeros((nNotas,3),dtype='float64')

    notaAnterior = midi.instruments[n_instrument].notes[0]

    # plt.figure(figsize=(12, 4))
    # plot_piano_roll(midi.instruments[n_instrumen], 24, 84)
    # plt.show()

    #print(notaAnterior)
    for i in range(0,nNotas):
        notaActual = midi.instruments[n_instrument].notes[i]
        dT = abs(notaActual.start - notaAnterior.start)
        # print("DT " ,dT)
        T = abs(notaActual.end - notaActual.start )
        # print("T " ,T)
        P = notaActual.pitch
        # print("p " ,P)
        #print(notaActual)
        notas[i][0] = dT
        notas[i][1] = aproximar(T/tempo)
        notas[i][2] = P

        notaAnterior = notaActual

    return notas, tempo

def reconstruirCancion(vector, tempo):
    i=0
    ac = 0
    print ("TEMPO", tempo)
    rec = pm.PrettyMIDI(initial_tempo=tempo*60)
    inst = pm.Instrument(program=1, is_drum=False, name='piano')
    rec.instruments.append(inst)

    for x in vector:
        ac+=x[0]*tempo
        p = x[2]
        start = ac
        # print(x[1])
        end = ac + aproximarInversa(x[1], tempo)
        #print ( p , start , end)

        inst.notes.append(pm.Note(100, int(p), start ,end))
        i+=1
    return rec
