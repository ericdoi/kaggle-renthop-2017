import argparse
import logging
import os
import pdb
import time

import numpy as np
import pandas as pd
from kaggler.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor

DEFAULT_DEPTH = 3
DEFAULT_LRATE = 0.3
DEFAULT_NROUNDS = 50  #250

numeric_features = ["bathrooms", "bedrooms", "latitude", "longitude"]

added_features = ["created_year"]  # , "created_month"]

categorical_features = []


def add_features(df):
    # convert the created column to datetime object so as to extract more features
    df["created"] = pd.to_datetime(df["created"])

    # Let us extract some features like year, month, day, hour from date columns #
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month


def replace_categorical_features(train_df, test_df):
    # Would have to save label encoders for each categorical feature to split train/test processing.
    # if train_df[f].dtype == 'object':
    cat_cols = categorical_features
    lbl = LabelEncoder(min_obs=10)
    cat_df = pd.concat([train_df, test_df], axis=0)[cat_cols]
    cat_df.ix[:, :] = lbl.fit_transform(cat_df.values)

    n_trn, n_tst = (train_df.shape[0], test_df.shape[0])
    train_df.ix[:, cat_cols] = cat_df.values[:n_trn, ]
    test_df.ix[:, cat_cols] = cat_df.values[n_trn:, ]


def load(train_path, test_path):
    with open(train_path) as train_file:
        with open(test_path) as test_file:
            train_df = pd.read_json(train_file)
            test_df = pd.read_json(test_file)
            print 'Train data:', train_df.shape
            print 'Test data:', test_df.shape
            return train_df, test_df


def initialize_logger(filename):
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=filename)

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.addHandler(handler)


def run(train_file, test_file, train_feature_file, test_feature_file, feature_map_file,
        depth, lrate, n_rounds, int_or_ohe):

    feat_name = os.path.splitext(
        os.path.splitext(os.path.basename(test_feature_file))[0]
    )[0]

    log_file = '{}.log'.format(feat_name)
    initialize_logger(log_file)

    logging.info('loading raw data')

    train_df, test_df = load(train_file, test_file)
    n_trn_rows = train_df.shape[0]
    n_tst_rows = test_df.shape[0]

    features_to_use = numeric_features + added_features + categorical_features

    logging.info('categorical: {}, numerical: {}, added: {}'.format(
        len(categorical_features), len(numeric_features), len(added_features)
    ))

    add_features(train_df)
    add_features(test_df)
    replace_categorical_features(train_df, test_df)

    full_df = pd.concat([train_df, test_df], axis=0)
    full_df['log_price'] = np.log(full_df['price'] + 1.0)

    X = full_df[features_to_use].values
    y = full_df['log_price'].values

    quantiles_to_check = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]

    # For each example, find out how many of the quantiles it is above
    decile_score = np.zeros(n_trn_rows + n_tst_rows)
    for col, alpha in enumerate(quantiles_to_check):
        print 'fitting quantile %.2f...' % alpha

        pred = predict_quantile(X, y, alpha, depth, lrate, n_rounds)
        indicator = full_df['log_price'] > pred  # pricier than the predicted quantile

        logging.info('Quantile %.2f:  fraction below = %.2f' % (alpha, 1.0 - np.mean(indicator)))

        decile_score += indicator

    # Save either as a single int feature or as one-hot
    if int_or_ohe == 'int':
        X_pq = decile_score.reshape(decile_score.size, 1)
        out_cols = ['price_quantile']
    elif int_or_ohe == 'ohe':
        ohe = OneHotEncoder(min_obs=10)
        X_pq = ohe.fit_transform(decile_score.reshape(decile_score.size, 1)).tocsr()
        out_cols = ['price_quantile_{}'.format(i) for i in range(X_pq.shape[1])]

    # Save feature mapping
    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(out_cols):
            f.write('{}\t{}\tq\n'.format(i, col))

    target_num_map = {'low': 0, 'medium': 1, 'high': 2}
    train_y = np.array(
        train_df['interest_level'].apply(lambda x: target_num_map[x])
    )

    test_y = np.zeros(n_tst_rows)

    logging.info('saving features')
    datasets.dump_svmlight_file(X_pq[:n_trn_rows,], train_y, train_feature_file)
    datasets.dump_svmlight_file(X_pq[n_trn_rows:,], test_y, test_feature_file)
    logging.info('Log file: %s' % log_file)


def predict_quantile(X, y, alpha, depth=DEFAULT_DEPTH, lrate=DEFAULT_LRATE, n=DEFAULT_NROUNDS, ):
    clf = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=n, max_depth=depth,
                                    learning_rate=lrate, min_samples_leaf=10, min_samples_split=10)
    clf.fit(X, y)
    preds = clf.predict(X)

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="path of training data file")
    parser.add_argument("--test-file", required=True, help="path of test data file")
    parser.add_argument("--train-feature-file", required=True, help="output path for train features")
    parser.add_argument("--test-feature-file", required=True, help="output path for test features")
    parser.add_argument('--feature-map-file', required=True, help="output path for feature indices")

    parser.add_argument("--depth", default=DEFAULT_DEPTH, type=int, help="max tree depth")
    parser.add_argument("--lrate", default=DEFAULT_LRATE, type=float, help="learning rate eta")
    parser.add_argument("--n-rounds", default=DEFAULT_NROUNDS, type=int, help="max num training rounds")
    parser.add_argument("--int-or-ohe", default="int", help="whether output should be int or one-hot")
    args = parser.parse_args()

    start = time.time()
    run(args.train_file, args.test_file,
        args.train_feature_file, args.test_feature_file,
        args.feature_map_file,
        args.depth, args.lrate, args.n_rounds, args.int_or_ohe)

    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))


if __name__ == "__main__":
    main()
