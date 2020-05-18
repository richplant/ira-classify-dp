from os import path
import tensorflow as tf
import keras_metrics as km
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


class CNNModel:
    def __init__(self, opts, output_size=1, filter_length=50, hidden_size=128, kernel_size=2):
        self.model = tf.keras.models.Sequential()
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
        self.model.add(tf.keras.Input(opts['input_shape']))
        self.model.add(tf.keras.layers.Reshape([1, opts['input_shape']]))
        self.model.add(tf.keras.layers.Conv1D(filter_length, kernel_size, padding='valid', activation='relu', strides=1, data_format='channels_first'))
        self.model.add(tf.keras.layers.GlobalMaxPooling1D(data_format='channels_first'))
        self.model.add(tf.keras.layers.Dense(hidden_size))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
        self.model.compile(loss=loss, optimizer='adam', metrics=['accuracy', km.categorical_precision(), km.categorical_recall(), km.categorical_f1_score()])


class LSTMModel:
    def __init__(self, opts, output_size=1, hidden_size=128):
        self.model = tf.keras.models.Sequential()
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(opts['input_shape']))
        self.model.add(tf.keras.layers.Reshape([1, opts['input_shape']]))
        self.model.add(tf.keras.layers.CuDNNLSTM(hidden_size))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
        self.model.compile(loss=loss, optimizer='adam', metrics=['accuracy', km.categorical_precision(), km.categorical_recall(), km.categorical_f1_score()])
        

class TFModels:
    def __init__(self, input_shape, batch_size=64, epochs=30):
        self.opts = {
            'input_shape': input_shape,
            'batch_size': batch_size,
            'epochs': epochs
        }
        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)
        self.models = {
            'CNN': CNNModel(self.opts).model,
            'LSTM': LSTMModel(self.opts).model,
        }

    def run_models(self, data, labels):
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels)
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=2)
        scores = {}
        for name, model in self.models.items():
            with tf.Session() as s:
                s.run(tf.global_variables_initializer())
                now = datetime.now()
                print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Fitting {name} model...")
                setattr(model, 'output_size', labels.shape[1])
                model_history = model.fit(train_data, train_labels,
                                        batch_size=self.opts['batch_size'],
                                        epochs=self.opts['epochs'],
                                        validation_data=(val_data, val_labels),
                                        callbacks=[self.early_stop])
                scores[name] = pd.DataFrame.from_dict(model_history.history)
        return scores

    def save_models(self, filepath):
        for name, model in self.models.items():
            model.save(filepath=path.join(filepath, f'{name}.h5'))

    def load_models(self, filepath):
        for name, model in self.models.items():
            if path.isfile(path.join(filepath, f'{name}.h5')):
                self.models[name] = tf.keras.models.load_model(path.join(filepath, f'{name}.h5'))

    def save_history(self, histories, outpath):
        report = {}
        for name, frame in histories.items():
            print(frame)
            report[name] = [frame['val_acc'].iloc[-1],
                            frame['val_precision'].iloc[-1],
                            frame['val_recall'].iloc[-1],
                            frame['val_f1_score'].iloc[-1]
            ]
            plt.figure()
            plt.plot(frame['val_loss'], label='Loss')
            plt.plot(frame['val_acc'], label='Accuracy')
            plt.plot(frame['val_precision'], label='Precision')
            plt.plot(frame['val_recall'], label='Recall')
            plt.plot(frame['val_f1_score'], label='F1 Score')
            plt.legend(loc='best')
            plt.title('Training history for {} model'.format(name))

            now = datetime.now()
            plt.savefig(path.join(outpath, f'{now.strftime("%Y-%m-%d")}-{name}-history.png'))
            frame.to_csv(path.join(outpath, f'{now.strftime("%Y-%m-%d")}-{name}-history.csv'), index=False)
        return pd.DataFrame.from_dict(report, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'f1'])

    def predict(self, data):
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict_classes(data, batch_size=self.opts['batch_size'])
        return results
