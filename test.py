import argparse
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from models import skmodels, tfmodels
import pandas as pd
import joblib


def save_maps(args, report):
    """
    Create and save heatmaps for confusion matrices
    :param report: dictionary of confusion matrices (keys = model names) 
    """
    for name, c_matrix in report.items():
        new_ax = plt.subplot(label='{}'.format(name))
        cm_plt = sns.heatmap(c_matrix, annot=True, fmt='d', ax=new_ax, cbar=None, cmap='Greens',
                             xticklabels=['none', 'astroturf'], yticklabels=['none', 'astroturf'])
        setattr(new_ax, 'xlabel', 'Predicted labels')
        setattr(new_ax, 'ylabel', 'True labels')
        cm_plt.title.set_text('Confusion matrix for {} model'.format(name))
        now = datetime.now()
        cm_plt.get_figure().savefig(args.o.joinpath('results').joinpath(f'{name}_cmatrix.png'))


def predict(args, data, labels, model_type='sk-models'):
    """
    Perform predictions with trained models
    :param data: numpy array of model input vectors
    :param labels: np array of true label values
    :param model_type: string name of model class to load
    :return: dictionary of predicted labels for test data (keys = model names, values = list of result metrics)
    """
    if model_type == 'sk-models':
        models = skmodels.SkModels()
    elif model_type == 'tf-models':
        models = tfmodels.TFModels(data.shape[1])
    else:
        raise KeyError('Model type not found.')

    models.load_models(args.o.joinpath('models'))
    predictions = models.predict(data)

    report = {}
    conf_report = {}
    
    for name, scores in predictions.items():
        print("Test scores for {} model, data dimensions: {}".format(name, data.shape))
        accuracy = accuracy_score(labels, scores)
        precision, recall, f_score, support = precision_recall_fscore_support(labels, scores, average='macro')
        roc_auc = roc_auc_score(labels, scores, average='macro')
        report[name] = [accuracy, precision, recall, f_score, roc_auc]
        conf_report[name] = confusion_matrix(labels, scores)

    save_maps(args, conf_report)

    pd_report = pd.DataFrame.from_dict(report, orient='index',
                                       columns=['accuracy', 'precision', 'recall', 'f1-score', 'roc_auc'])
    now = datetime.now()
    pd_report.to_csv(args.o.joinpath('results').joinpath(f'{now.strftime("%Y-%m-%d")}-{model_type}-results.csv'), index=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Get model predictions on test data and evaluate.')
    parser.add_argument('--input', dest="input_dir", help="Data input folder", required=True)
    parser.add_argument('--output', dest="output_dir", help="Experimental output folder", required=True)
    args = parser.parse_args()

    args.i = Path.cwd().joinpath(args.input_dir)
    args.o = Path.cwd().joinpath(args.output_dir)
    Path(args.o.joinpath('results')).mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Loading data...")
    data = joblib.load(args.i.joinpath('test').joinpath('data.gz'))
    labels = joblib.load(args.i.joinpath('test').joinpath('labels.gz'))
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Data loaded.")

    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Making predictions...")
    predict(args, data=data, labels=labels, model_type='sk-models')
    predict(args, data=data, labels=labels, model_type='tf-models')
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Testing complete.")


if __name__ == "__main__":
    main()