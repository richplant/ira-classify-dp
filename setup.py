import argparse
from preproc import dbsetup, frameset, preprocess
import os
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
import joblib
from os import listdir


def import_files(args):
    """
    Import data from files into mongodb and create merged collections
    """
    importer = dbsetup.Importer('astroturf')
    print("Importing Twitter set...")
    importer.clear(col_name='ira_tweets')
    importer.import_tweets_from_file(col_name='ira_tweets', filename=f'{args.input_dir}/ira_tweets_csv_unhashed.csv')
    importer.clear(col_name='recent_tweets')
    for file in listdir(f'{args.input_dir}/clean'):
        importer.import_tweets_from_file(col_name='recent_tweets', filename=f'{args.input_dir}/clean/' + file, json_type=True, borked_json=True)
    importer.clear(col_name='merged_tweets')
    importer.merge_tweets(merge_col='merged_tweets', col_1='ira_tweets', col_2='recent_tweets')
    print('Finished importing.')


def load(collection):
    """
    Load and format tweet data as Pandas dataframe
    :param collection: Collection name to read from
    :return: Pandas Dataframe
    """
    print("Reading to Dataframe...")
    framer = frameset.Framer('astroturf')
    df = framer.get_frame(collection)
    print("Dataframe created.")
    return df


def process(df, args):
    """
    Apply data transformations
    :type df: Pandas dataframe
    """
    print("Applying transformations...")
    preprocessor = preprocess.PreProcessor()
    data_arr, labels = preprocessor.transform(df)
    print("Transformation produced a: {}".format(type(data_arr)))
    print("With shape: {}".format(data_arr.shape))

    return data_arr, labels


def main():
    parser = argparse.ArgumentParser(description='This script loads and samples the dataset from file archives.')
    parser.add_argument('--output', dest="output_dir", help="Data destination folder")
    parser.add_argument('--input', dest="input_dir", help="Data input folder")
    parser.add_argument('--frac', dest="frac", type=float, help="Fraction of set to sample (as float)")

    args = parser.parse_args()

    #import_files(args)

    df = load('merged_tweets')
    df.to_pickle(os.path.join(args.output_dir, 'data_raw.pkl'))

    sample_len = int(len(df) * args.frac // 2)
    ndf = df.groupby('label').apply(lambda x: x.sample(n=sample_len)).reset_index(drop=True)
    data, labels = process(ndf, args)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=1)

    joblib.dump(X_train, os.path.join(args.output_dir, 'train/data.gz'), compress=3)
    joblib.dump(y_train, os.path.join(args.output_dir, 'train/labels.gz'), compress=3)

    #sparse.save_npz(os.path.join(args.output_dir, 'test/data.npz'), X_test)
    joblib.dump(X_test, os.path.join(args.output_dir, 'test/data.gz'), compress=3)
    joblib.dump(y_test, os.path.join(args.output_dir, 'test/labels.gz'), compress=3)


if __name__ == "__main__":
    main()