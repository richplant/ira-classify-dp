from os import path
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Reshape, Bidirectional, LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

class Metrics(Callback):
    def __init__(self):
        super(Metrics, self).__init__()
        self.val_accs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_roc_aucs = []

    def on_train_begin(self, logs=None):
        self.val_accs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_roc_aucs = []

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_acc = accuracy_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        _val_auc = roc_auc_score(val_targ, val_predict, average="macro")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_accs.append(_val_acc)
        self.val_roc_aucs.append(_val_auc)
        logs['accuracy'] = _val_acc
        logs['f1-score'] = _val_f1
        logs['recall'] = _val_recall
        logs['precision'] = _val_precision
        logs['roc_auc'] = _val_auc
        print("- val_f1: %f - val_precision: %f - val_recall %f - val_roc_auc %f"
              % (_val_f1, _val_precision, _val_recall, _val_auc))
        return


class CNNModel:
    def __init__(self, input_length, output_size=1, filter_length=50, hidden_size=128, kernel_size=2):
        self.model = Sequential()
        self.model.add(Reshape((1, input_length), input_shape=(input_length, )))
        self.model.add(Conv1D(filter_length, kernel_size, padding='valid', activation='relu', strides=1,
                              data_format='channels_first'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(hidden_size))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_size, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


class LSTMModel:
    def __init__(self, input_length, output_size=1, hidden_size=128):
        self.model = Sequential()
        self.model.add(Reshape((1, input_length), input_shape=(input_length,)))
        self.model.add(Bidirectional(LSTM(hidden_size)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(output_size, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


class TFModels:
    def __init__(self, input_length, batch_size=32, epochs=30):
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics = Metrics()
        self.early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
        self.models = {
            'CNN': CNNModel(input_length).model,
            'LSTM': LSTMModel(input_length).model,
        }

    def run_models(self, data, labels):
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels)
        scores = {}
        for name, model in self.models.items():
            now = datetime.now()
            print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Fitting {name} model...")
            setattr(model, 'output_size', labels.shape[1])
            model_history = model.fit(train_data, train_labels,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=(val_data, val_labels),
                                      callbacks=[self.metrics, self.early_stop])
            scores[name] = pd.DataFrame.from_dict(model_history.history)
        return scores

    def save_models(self, filepath):
        for name, model in self.models.items():
            model.save(filepath=path.join(filepath, f'{name}.h5'))

    def load_models(self, filepath):
        for name, model in self.models.items():
            if path.isfile(path.join(filepath, f'{name}.h5')):
                self.models[name] = load_model(path.join(filepath, f'{name}.h5'))

    def save_history(self, histories, outpath):
        report = {}
        for name, frame in histories.items():
            report[name] = [frame['accuracy'].iloc[-1],
                            frame['precision'].iloc[-1],
                            frame['recall'].iloc[-1],
                            frame['f1-score'].iloc[-1],
                            frame['roc_auc'].iloc[-1]]
            plt.figure()
            plt.plot(frame['accuracy'], label='Accuracy')
            plt.plot(frame['precision'], label='Precision')
            plt.plot(frame['recall'], label='Recall')
            plt.plot(frame['f1-score'], label='F1 Score')
            plt.plot(frame['roc_auc'], label='ROC AUC')
            plt.legend(loc='best')
            plt.title('Training history for {} model'.format(name))

            now = datetime.now()
            plt.savefig(path.join(outpath, f'{now.strftime("%Y-%m-%d")}-{name}-history.png'))
            frame.to_csv(path.join(outpath, f'{now.strftime("%Y-%m-%d")}-{name}-history.csv'), index=False)
        return pd.DataFrame.from_dict(report, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'f1', 'ROC AUC'])

    def predict(self, data):
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict_classes(data, batch_size=self.batch_size)
        return results
