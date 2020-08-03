"""
This file can be used to try a live prediction. 
"""
import pyaudio
import wave
import keras
import librosa
import numpy as np

def recording():
        filename = "recorded.wav"
        chunk = 1024
# sample format
        FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
        channels = 1

        sample_rate = 44100
        record_seconds = 5
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=chunk)
        frames = []
        print("Recording...")
        for i in range(int(44100 / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
        print("Finished recording.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(filename, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()

class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file
        self.path = '/root/Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print( "Prediction is", " ", self.convert_class_to_emotion(predictions))

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'ok',
                            '1': 'ok',
                            '2': 'ok',
                            '3': 'ok',
                            '4': 'Distress signal',	#angry and fearful signal are distressed
                            '5': 'Distress signal',
                            '6': 'ok',
                            '7': 'ok'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label
    


if __name__ == '__main__':
    recording()
    live_prediction = LivePredictions(file='321.wav')
    live_prediction.make_predictions()
