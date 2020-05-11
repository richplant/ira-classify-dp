import tensorflow as tf
from tensorflow.python.client import device_lib 
import joblib
from models import skmodels
from models import tfmodels
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd


def train(args, data, labels, model_type='sk-models'):
    """
    Train and validate model set
    :param model_type: string name of model class to load
    :param data: np array of data input vectors
    :param labels: np array of label values
    """
    if model_type == 'sk-models':
        models = skmodels.SkModels()
        scores = models.run_models(data, labels)
        models.save_models(args.o.joinpath('models'))
        print(scores)
        now = datetime.now()
        scores.to_csv(args.o.joinpath('training').joinpath(f'{now.strftime("%Y-%m-%d")}-sk-models.csv'), index=True)

    elif model_type == 'tf-models':
        models = tfmodels.TFModels(data.shape[1])
        scores = models.run_models(data, labels)
        models.save_models(args.o.joinpath('models'))
        report = models.save_history(scores, args.o.joinpath('training'))
        print(report)
        now = datetime.now()
        report.to_csv(args.o.joinpath('training').joinpath(f'{now.strftime("%Y-%m-%d")}-tf-models.csv'), index=True)

    else:
        raise KeyError('Model type not found.')


def main():
    parser = argparse.ArgumentParser(description='Trains models on dataset.')
    parser.add_argument('--input', dest="input_dir", help="Data input folder", required=True)
    parser.add_argument('--output', dest="output_dir", help="Experimental output folder", required=True)
    args = parser.parse_args()

    args.i = Path.cwd().joinpath(args.input_dir)
    args.o = Path.cwd().joinpath(args.output_dir)
    Path(args.o.joinpath('models')).mkdir(parents=True, exist_ok=True)
    Path(args.o.joinpath('training')).mkdir(parents=True, exist_ok=True)

    gpus = [device for device in device_lib.list_local_devices() if device.device_type == 'GPU']
    print("Num GPUs Available: ", len(gpus))
    if len(gpus) < 1:
        print('GPU required to run.')
        return False
    for device in gpus:
        print(device.physical_device_desc)

    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Loading data...")
    data = joblib.load(args.i.joinpath('train').joinpath('data.gz'))
    labels = joblib.load(args.i.joinpath('train').joinpath('labels.gz'))
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Data loaded.")

    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Running model training...")
    train(args, data=data, labels=labels, model_type='sk-models')
    train(args, data=data, labels=labels, model_type='tf-models')
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Training complete.")


if __name__ == "__main__":
    main()