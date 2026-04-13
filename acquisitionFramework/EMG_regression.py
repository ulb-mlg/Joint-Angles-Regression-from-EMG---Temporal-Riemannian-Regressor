import os
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF Lite delegate / feedback warnings
print("importing tensorflow ...")
import tensorflow as tf
from models import createRNN

from absl import logging as loggingabsl
loggingabsl.set_verbosity(loggingabsl.ERROR)  # suppress Mediapipe feedback manager logs

import threading
from keras.src.callbacks import EarlyStopping
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from rbcx.handtracker.mediapipe import MediaPipeHandTracker
from joblib import dump, load
from scipy.signal import butter, sosfilt
import config
from EMG import MindRoveEMG, EMG_SFREQ, EMG_N_CHANNEL

import logging
logging.getLogger("data_logger").setLevel(logging.CRITICAL)



EMG_WINDOW_LENGTH = config.EMG_WINDOW_LENGTH
EMG_WINDOW_STEP = config.EMG_WINDOW_STEP
EMG_SEQUENCE_LENGTH = config.EMG_SEQUENCE_LENGTH
freqBands = config.FREQ_BANDS


def bandpass_filter(data, sos):
    return sosfilt(sos, data, axis=0)


def extractSamples(emg, labels, ts):
    x_emg = []
    y = []
    for j, t in enumerate(ts):
        if t > len(emg):
            break
        if t > EMG_WINDOW_LENGTH + EMG_WINDOW_STEP * EMG_SEQUENCE_LENGTH:
            sequence_emg = []
            for i in range(EMG_SEQUENCE_LENGTH):
                sequence_emg.append(emg[t - EMG_WINDOW_LENGTH - EMG_WINDOW_STEP * i:t - EMG_WINDOW_STEP * i])
            x_emg.append(sequence_emg)
            if labels is not None:
                y.append(labels[j])
    if labels is None:
        return np.array(x_emg).transpose((0, 1, 3, 2))[:, ::-1], None
    return np.array(x_emg).transpose((0, 1, 3, 2))[:, ::-1], np.array(y)


def labelInterpolation(emg, label, yts):
    labelForFif = np.zeros((emg.shape[0], 15)) * np.nan
    labelForFif[yts[yts < len(labelForFif)]] = label[yts < len(labelForFif)]
    i = 0
    while i < len(labelForFif):
        if not np.isnan(labelForFif[i, 0]):
            j = i + 1
            while np.isnan(labelForFif[j, 0]) and j < len(labelForFif) - 1:
                j += 1
            if j > i + 1:
                dist = j - i
                diff = labelForFif[j] - labelForFif[i]
                for k in range(i + 1, j):
                    labelForFif[k] = labelForFif[i] + (k - i) * diff / dist
        i += 1
    select = np.logical_not(np.isnan(labelForFif[:, 0]))
    return emg[select].T, labelForFif[select].T


def extractFeatures_10fps(emg, label):
    size=EMG_WINDOW_LENGTH
    step=EMG_WINDOW_STEP
    seq=EMG_SEQUENCE_LENGTH

    ts = np.arange(1000, len(emg)-1000, step)

    x_emg = []
    y = []
    for t in ts:
        x_emg.append(emg[t - size+1:t+1].transpose())
        y.append(label[t])
    x_emg = np.array(x_emg)

    print(" - computing covariance matrices")
    cmtsExtractor = Pipeline([('cov', Covariances(estimator='oas')), ('ts', TangentSpace('riemann'))])
    cmtsExtractor.fit(x_emg)
    features = cmtsExtractor.transform(x_emg)

    features_sequence = []
    Y = []
    for i in range(seq, len(x_emg)-1):
        features_sequence.append(features[i-seq+1:i+1])
        Y.append(y[i])

    features_sequence = np.array(features_sequence)
    y = np.array(Y)

    feature_concatenated = np.concatenate(features_sequence.transpose((1, 0, 2)), axis=1)
    return features_sequence, feature_concatenated, y, cmtsExtractor



class EMG_regressor:
    def __init__(self, handTracker : MediaPipeHandTracker):
        self.handTracker = handTracker
        self.pred = np.zeros(15)

        self.stopProgram = False
        self.retrain = False
        self.busy = False

        self.emg = []
        self.label = []
        self.labelTs = []  # times of labels in EMG
        self.trainTimes = []
        self.shownPred = []

    @property
    def show_window(self):
        return self.handTracker.show_window

    @show_window.setter
    def show_window(self, v):
        self.handTracker.show_window = v

    def poll_gui(self):
        self.handTracker.poll_gui()

    def userInputThread(self):
        time.sleep(10)
        while not self.stopProgram:
            if not self.busy:
                message = input("train (t), save(s) : ")
                if message == "t":
                    self.retrain = True
                    self.busy = True
                elif message == "s":
                    sizeLabel = len(self.labelTs)
                    sizeEMG = len(self.emg)
                    np.save("savedData/EMG_data.npy", self.emg[:sizeEMG])
                    np.save("savedData/label_data.npy", self.label[:sizeLabel])
                    np.save("savedData/label_ts_data.npy", self.labelTs[:sizeLabel])
                    np.save("savedData/shown_pred_data.npy", self.shownPred[:sizeLabel])
                    np.save("savedData/train_times.npy", self.trainTimes)
            else:
                time.sleep(1)


    def stop(self):
        self.stopProgram = True


    def start(self):
        self.handTracker.start()
        self.mindrove = MindRoveEMG()

        self.model = None
        self.input_details = None
        self.output_details = None

        self.sos = [butter(4, freqBands[i], btype='bandpass', fs=EMG_SFREQ, output='sos') for i in range(len(freqBands))]

        self.scalers = [None] * 15
        self.emgScalers = [[None] * EMG_N_CHANNEL for _ in range(len(freqBands))]
        self.cmtsExtractor = [None] * len(freqBands)

        threadReceiver = threading.Thread(target=self.userInputThread)
        threadReceiver.start()

        if config.PRELOAD_EMG_MODEL:
            self.model = tf.lite.Interpreter(model_path='savedModel/model.tflite')
            self.model.allocate_tensors()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()

            self.scalers = np.load("savedModel/labelScaler.npy", allow_pickle=True)
            self.emgScalers = np.load("savedModel/emgScaler.npy", allow_pickle=True)
            for b in range(len(freqBands)):
                self.cmtsExtractor[b] = load(f"savedModel/pyriemann_pipeline_band_{b}.joblib")


    def __step(self):
        data_emg = self.mindrove.getEMG()
        mediapipe_joints_angles = self.handTracker.get_mediapipe_angles()

        self.emg.extend(data_emg.T)
        self.labelTs.append(len(self.emg))
        self.label.append(mediapipe_joints_angles)
        pred = self.label[-1]

        # angles prediction from EMG
        if self.model is not None and len(self.emg) >= EMG_WINDOW_STEP * EMG_SEQUENCE_LENGTH + EMG_WINDOW_LENGTH + 200:
            emgToExtract = np.array(self.emg[-EMG_WINDOW_LENGTH - EMG_WINDOW_STEP * EMG_SEQUENCE_LENGTH - 100:])

            x = []
            for b, bands in enumerate(freqBands):
                emgFiltered = bandpass_filter(deepcopy(emgToExtract), self.sos[b])

                for c in range(EMG_N_CHANNEL):
                    emgFiltered[:, c] -= self.emgScalers[b][c][0]
                    emgFiltered[:, c] /= self.emgScalers[b][c][1]
                X_emg, _ = extractSamples(emgFiltered, None, [len(emgFiltered)])

                features = []
                for i in range(EMG_SEQUENCE_LENGTH):
                    x_emg = X_emg[:, i]
                    featuresI = self.cmtsExtractor[b].transform(x_emg)
                    features.append(featuresI)
                x_emg = np.array(features).transpose((1, 0, 2))
                x.append(x_emg.transpose((2, 1, 0)))
            x = np.concatenate(x).transpose((2, 1, 0))

            self.model.set_tensor(self.input_details[0]['index'], x.astype(np.float32))
            self.model.invoke()
            pred = self.model.get_tensor(self.output_details[0]['index'])[0]

            for i in range(15):
                pred[i] = self.scalers[i].inverse_transform([[pred[i]]])[0][0]

        self.pred = pred

        self.shownPred.append(pred)

        # model retraining
        if self.retrain:
            self.trainTimes.append(len(self.label))

            self.retrain = False
            print("Train model")
            print(" - filtering signal")

            x = []
            y = None
            for b, bands in enumerate(freqBands):
                emgFiltered = np.array(deepcopy(self.emg))
                emgToExtract = bandpass_filter(emgFiltered, self.sos[b])

                print(" - scaling signal")
                for c in range(EMG_N_CHANNEL):
                    self.emgScalers[b][c] = (np.mean(emgToExtract[:, c]), np.mean(np.abs(emgToExtract[:, c])))
                    emgToExtract[:, c] -= self.emgScalers[b][c][0]
                    emgToExtract[:, c] /= self.emgScalers[b][c][1]

                print(" - extracting samples")
                emg2, y2 = labelInterpolation(emgToExtract, np.array(self.label), np.array(self.labelTs))
                if y is None:
                    x_, _, y, cmtsExtractor_ = extractFeatures_10fps(emg2.T, y2.T)
                else:
                    x_, _, _, cmtsExtractor_ = extractFeatures_10fps(emg2.T, y2.T)
                x.append(x_.transpose((2, 1, 0)))
                self.cmtsExtractor[b] = cmtsExtractor_
            x = np.concatenate(x)
            x = x.transpose((2, 1, 0))

            print(" - scaling samples")
            for i in range(15):
                self.scalers[i] = StandardScaler().fit(y[:, i].reshape(-1, 1))
                y[:, i] = self.scalers[i].transform(y[:, i].reshape(-1, 1))[:, 0]

            print(" - training model")
            b = int(len(x) * 0.95)
            trainX = x[:b]
            trainY = y[:b]
            valX = x[b:]
            valY = y[b:]

            self.model = createRNN(output_dim=15, input_shape=(x.shape[1], x.shape[2]))
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=False,
                verbose=1
            )
            self.model.fit( trainX,
                            trainY,
                            epochs=100,
                            batch_size=128,
                            validation_data=(valX, valY),
                            callbacks=[early_stop])

            print(" - save model")
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
            with open('savedModel/model.tflite', 'wb') as f:
                f.write(tflite_model)
            self.model = tf.lite.Interpreter(model_path='savedModel/model.tflite')
            self.model.allocate_tensors()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()

            np.save("savedModel/labelScaler.npy", np.array(self.scalers))
            np.save("savedModel/emgScaler.npy", np.array(self.emgScalers))
            for b in range(len(freqBands)):
                dump(self.cmtsExtractor[b], f"savedModel/pyriemann_pipeline_band_{b}.joblib")

            self.mindrove.clearBuffer()   # delete all EMG collected during model training

            self.busy = False

            print("starting in 5 seconds")
            time.sleep(5)


    def get_hand_state(self, side="Right"):
        if side == "Left":
            return None
        self.__step()
        return {"angles_list":self.pred, "landmarks":None}
