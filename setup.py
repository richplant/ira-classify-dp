import argparse
from preproc import dbsetup, frameset, preprocess
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
import joblib
from os import listdir
from pathlib import Path
from datetime import datetime


def import_files(args):
    """
    Import data from files into mongodb and create merged collections
    """
    importer = dbsetup.Importer('astroturf')
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Importing Twitter set...")
    # importer.clear(col_name='ira_tweets')
    # importer.import_tweets_from_file(col_name='ira_tweets', filename=args.i.joinpath('ira_tweets_csv_unhashed.csv'))
    # importer.clear(col_name='recent_tweets')
    # for file in listdir(f'{args.input_dir}/clean'):
    #     importer.import_tweets_from_file(col_name='recent_tweets', filename=args.i.joinpath('clean').joinpath(file), json_type=True, borked_json=True)
    importer.clear(col_name='merged_tweets')
    importer.merge_tweets(merge_col='merged_tweets', col_1='ira_tweets', col_2='recent_tweets')
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Finished importing.")


def load(collection):
    """
    Load and format tweet data as Pandas dataframe
    :param collection: Collection name to read from
    :return: Pandas Dataframe
    """
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Reading to Dataframe...")
    framer = frameset.Framer('astroturf')
    df = framer.get_frame(collection)
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Dataframe created.")
    return df


def process(df, args):
    """
    Apply data transformations
    :type df: Pandas dataframe
    """
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Applying transformations...")
    preprocessor = preprocess.PreProcessor()
    data_arr, labels = preprocessor.transform(df)
    now = datetime.now()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} Transformation produced a: {type(data_arr)}")
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} With shape: {data_arr.shape}")

    return data_arr, labels


def main():
    parser = argparse.ArgumentParser(description='This script loads and samples the dataset from file archives.')
    parser.add_argument('--output', dest="output_dir", help="Data destination folder", required=True)
    parser.add_argument('--input', dest="input_dir", help="Data input folder", required=True)
    parser.add_argument('--frac', dest="frac", type=float, help="Fraction of set to sample (as float)", required=True)

    args = parser.parse_args()

    args.i = Path.cwd().joinpath(args.input_dir)
    args.o = Path.cwd().joinpath(args.output_dir)

    Path(args.o.joinpath('train')).mkdir(parents=True, exist_ok=True)
    Path(args.o.joinpath('test')).mkdir(parents=True, exist_ok=True)

    import_files(args)

    df = load('merged_tweets')
    df.to_pickle(args.o.joinpath('data_raw.pkl'))

    sample_len = int(len(df) * args.frac // 2)
    ndf = df.groupby('label').apply(lambda x: x.sample(n=sample_len)).reset_index(drop=True)
    print(ndf.head())
    print(f'Instances per label group: {ndf.groupby("label")["created_at"].count()}')
    data, labels = process(ndf, args)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=1)

    joblib.dump(X_train, args.o.joinpath('train/data.gz'), compress=3)
    joblib.dump(y_train, args.o.joinpath('train/labels.gz'), compress=3)

    joblib.dump(X_test, args.o.joinpath('test/data.gz'), compress=3)
    joblib.dump(y_test, args.o.joinpath('test/labels.gz'), compress=3)


if __name__ == "__main__":
    main()