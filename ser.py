"""
This file can be used to try a live prediction. 
"""

import keras
import librosa
import numpy as np
import recording
import sys, select, os
import msvcrt, time


def ser(file):
    path='Emotion_Voice_Detection_Model.h5'
    loaded_model = keras.models.load_model(path)
    data, sampling_rate = librosa.load(file)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    predictions = loaded_model.predict_classes(x)
    print( "Prediction is", " ",convert_class_to_emotion(predictions))
def convert_class_to_emotion(pred):
    label_conversion = {'0': 'ok',
                            '1': 'ok',
                            '2': 'ok',
                            '3': 'ok',
                            '4': 'Distress',	#angry and fearful signal are distressed
                            '5': 'Distress',
                            '6': 'ok',
                            '7': 'ok'}

    for key, value in label_conversion.items():
        if int(key) == pred:
            label = value
    return label
def final(*args):
    if(args[0]==0):
        '''
        for unix start
        i = 0
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            recording.record()
            ser('output.wav')
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = raw_input()
                break
            i += 1
            #for unix end
    ''' 
     #for windows start
        i = 0
        while True:
            i = i + 1
            recording.record()
            ser('output.wav')
            if msvcrt.kbhit():
                if msvcrt.getwche() == '\r':
                    break
            time.sleep(0.1)
        print(i)
        #for windows end
    else:
        ser(args[0])
if __name__ == '__main__':
    final('321.wav')
