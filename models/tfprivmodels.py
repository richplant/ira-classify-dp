import tensorflow as tf
import keras_metrics as km
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from os import path


class pCNNModel:
    def __init__(self, opts, output_size=1, filter_length=50, hidden_size=128, kernel_size=2):
        self.optimizer = DPAdamGaussianOptimizer(
            l2_norm_clip=opts['l2_norm_clip'],
            noise_multiplier=opts['noise_multiplier'],
            num_microbatches=opts['microbatches'],
            learning_rate=opts['learning_rate']
        )
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv1D(filter_length, kernel_size, padding='valid', activation='relu', strides=1, data_format='channels_first', input_shape=opts['input_shape']))
        self.model.add(tf.keras.layers.GlobalMaxPooling1D())
        self.model.add(tf.keras.layers.Dense(hidden_size))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(opts['batch_size'], activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy', km.precision(), km.recall(), km.f1_score()])
        print(self.model.summary())


class pLSTMModel:
    def __init__(self, opts, output_size=1, hidden_size=128):
        self.optimizer = DPAdamGaussianOptimizer(
            l2_norm_clip=opts['l2_norm_clip'],
            noise_multiplier=opts['noise_multiplier'],
            num_microbatches=opts['microbatches'],
            learning_rate=opts['learning_rate']
        )
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size), input_shape=opts['input_shape']))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(opts['batch_size'], activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy', km.precision(), km.recall(), km.f1_score()])
        print(self.model.summary())


class TFPrivModels:
    def __init__(self, input_shape, batch_size=1, epochs=30): # batch_size must be 1 since tensorflow_privacy expects dimension % batch_size == 0
        self.opts = {
            'input_shape': input_shape,
            'learning_rate': 0.15,
            'noise_multiplier': 1.1,
            'l2_norm_clip': 1.0,
            'microbatches': batch_size,
            'batch_size': batch_size,
            'epochs': epochs
        }
        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
        self.models = {
            'pCNN': pCNNModel(self.opts).model,
            'pLSTM': pLSTMModel(self.opts).model,
        }

    def run_models(self, data, labels):
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels)
        scores = {}
        for name, model in self.models.items():
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
            results[name] = model.predict_classes(data, batch_size=self.opts['batch_size'])
        return results