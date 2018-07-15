import pretty_midi as pm
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def plot_piano_roll(midi, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(midi.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pm.note_number_to_hz(start_pitch))

duracionSet = [0.0625, 0.125, 0.25, 0.5,1, 2,4]



def aproximar(d):
    aprox = 0.0
    dist = abs(d - duracionSet[0])
    for i in range(1, len(duracionSet)):
        if abs(duracionSet[i] - d ) < dist:
            dist = abs(duracionSet[i] - d )
            aprox = i
    return aprox




def get_vector(file,n_instrument=0):
    midi = pm.PrettyMIDI(file)
    tempo = float(midi.estimate_tempo()/60)
    print('tempo ',midi.estimate_tempo()) ,
    print('instrumentos' , len(midi.instruments))

    nNotas = len(midi.instruments[n_instrument].notes)
    notas = np.zeros((nNotas,3),dtype='float64')

    notaAnterior = midi.instruments[n_instrument].notes[0]

    plt.figure(figsize=(12, 4))
    plot_piano_roll(midi.instruments[n_instrument], 24, 84)
    plt.show()

    #print(notaAnterior)
    for i in range(1,nNotas):
        notaActual = midi.instruments[n_instrument].notes[i]
        dT = abs(notaActual.start - notaAnterior.start)
        print("DT " ,dT)
        T = abs(notaActual.end - notaActual.start )
        print("T " ,T)
        P = notaActual.pitch
        print("p " ,P)
        #print(notaActual)
        notas[i-1][0] = dT/tempo
        notas[i-1][1] = aproximar(T/tempo)#notaActual.end#T#  /tempo#a
        notas[i-1][2] = P

        notaAnterior = notaActual

    return notas


for x in get_vector('midis/himno.mid',0):
    print (x)
