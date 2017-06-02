import argparse
import logging
import time

import numpy as np
import pandas as pd
from kaggler.preprocessing import OneHotEncoder, Normalizer
from scipy import sparse
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer


numeric_features = ["bathrooms", "bedrooms", "price", "log_price", "num_photos", "num_features",
                    "log_num_features", "has_long_feature", "has_all_caps_feature", "num_description_words",
                    "log_num_description_words"]

categorical_features = ["display_address", "manager_id", "building_id", "street_address", "created_year",
                        "created_month", "created_day", "created_hour"]

LONG_FEAT_CHARS = 27


def add_features(df):
    df["log_price"] = np.log(df["price"] + 1)

    # count of photos #
    df["num_photos"] = df["photos"].apply(len)

    # count of "features" #
    df["num_features"] = df["features"].apply(len)
    df["log_num_features"] = np.log(df["num_features"] + 1)

    df["has_long_feature"] = df["features"].apply(
        lambda lst: 1 * (max((len(tag) for tag in lst + [""])) >= LONG_FEAT_CHARS)
    )
    df["has_all_caps_feature"] = df["features"].apply(
        lambda lst: 1 * any([len(tag) >= 5 and tag == tag.upper() for tag in lst])
    )

    # count of words present in description column #
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["log_num_description_words"] = np.log(df["num_description_words"] + 1)

    # convert the created column to datetime object so as to extract more features
    df["created"] = pd.to_datetime(df["created"])

    # Let us extract some features like year, month, day, hour from date columns #
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour


def normalize_features(df, cols):
    nm = Normalizer()
    df.ix[:, cols] = nm.fit_transform(df[cols].values)


def make_categorical_features(df, cols):
    ohe = OneHotEncoder(min_obs=10)
    X_ohe = ohe.fit_transform(df[cols].values)
    ohe_cols = ['ohe{}'.format(i) for i in range(X_ohe.shape[1])]
    return X_ohe, ohe_cols


def make_text_features(df):
    df['features'] = df["features"].apply(
        lambda lst: " ".join(["_".join(tag.split()) for tag in lst])
    )
    print(df["features"].head())
    ctr = CountVectorizer(stop_words='english', max_features=200)
    X_text = ctr.fit_transform(df["features"])

    tag_names = ['tag_%s' % tag for tag in ctr.get_feature_names()]
    return X_text, tag_names


def load(train_path, test_path):
    with open(train_path) as train_file:
        with open(test_path) as test_file:
            train_df = pd.read_json(train_file)
            test_df = pd.read_json(test_file)
            print 'Train data:', train_df.shape
            print 'Test data:', test_df.shape
            return train_df, test_df


def run(train_file, test_file, train_feature_file, test_feature_file, feature_map_file):
    logging.info('loading raw data')

    train_df, test_df = load(train_file, test_file)
    n_trn_rows = train_df.shape[0]
    n_tst_rows = test_df.shape[0]

    logging.info('categorical: {}, numerical: {}'.format(
        len(categorical_features), len(numeric_features)
    ))

    full_df = pd.concat([train_df, test_df], axis=0)
    add_features(full_df)
    normalize_features(full_df, numeric_features)
    X_ohe, ohe_cols = make_categorical_features(full_df, categorical_features)
    X_text, tag_names = make_text_features(full_df)

    full_X = sparse.hstack([full_df[numeric_features], X_ohe, X_text]).tocsr()


    # Save feature mapping
    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(numeric_features + ohe_cols + tag_names):
            f.write('{}\t{}\tq\n'.format(i, col))

    target_num_map = {'low': 0, 'medium': 1, 'high': 2}
    train_y = np.array(
        train_df['interest_level'].apply(lambda x: target_num_map[x])
    )

    test_y = np.zeros(n_tst_rows)

    logging.info('saving features')
    datasets.dump_svmlight_file(full_X[:n_trn_rows, ], train_y, train_feature_file)
    datasets.dump_svmlight_file(full_X[n_trn_rows:, ], test_y, test_feature_file)


def main():
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="path of training data file")
    parser.add_argument("--test-file", required=True, help="path of test data file")
    parser.add_argument("--train-feature-file", required=True, help="output path for train features")
    parser.add_argument("--test-feature-file", required=True, help="output path for test features")
    parser.add_argument('--feature-map-file', required=True, help="output path for feature indices")
    args = parser.parse_args()

    start = time.time()
    run(args.train_file, args.test_file,
        args.train_feature_file, args.test_feature_file,
        args.feature_map_file)

    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))


if __name__ == "__main__":
    main()
