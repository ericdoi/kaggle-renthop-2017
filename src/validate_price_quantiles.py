"""
Get CV performance metrics for quantile price model, to compare parameters
"""

import argparse
import logging
import pdb
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from kaggler.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

DEFAULT_FOLDS = 3
CV_SEED = 2017
XG_SEED = 0
FMAP_FILE = 'validate_price_quantiles.fmap'
FEAT_IMP_FILE = 'validate_price_quantiles.imp'
EARLY_STOPPING = 20

numeric_features = ["bathrooms", "bedrooms", "latitude", "longitude"]

added_features = ["created_year", "created_month"]

categorical_features = []


def add_features(df):
    # convert the created column to datetime object so as to extract more features
    df["created"] = pd.to_datetime(df["created"])

    # Let us extract some features like year, month, day, hour from date columns #
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month


def replace_categorical_features(train_df):
    # Would have to save label encoders for each categorical feature to split train/test processing.
    # if train_df[f].dtype == 'object':
    cat_cols = categorical_features
    lbl = LabelEncoder(min_obs=10)
    train_df.ix[:, cat_cols] = lbl.fit_transform(train_df[cat_cols])


def load(train_path):
    with open(train_path) as train_file:
        train_df = pd.read_json(train_file)
        print 'Train data:', train_df.shape
        return train_df


def initialize_logger(filename):
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=filename)

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.addHandler(handler)


def run(train_file, depth, lrate, n_rounds, n_folds=DEFAULT_FOLDS):
    feat_name = 'validate_pq_%s_%s_%s' % (depth, lrate, n_rounds)
    log_file = '{}.log'.format(feat_name)

    initialize_logger(log_file)

    param_dict = {
        'objective': 'reg:linear',
        'eta': lrate,
        'max_depth': depth,
        'silent': 1,
        #'eval_metric': "mae",
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': XG_SEED
    }

    params = list(param_dict.items())

    logging.info('loading raw data')

    df = load(train_file)

    features_to_use = numeric_features + added_features + categorical_features

    # Save feature mapping
    with open(FMAP_FILE, 'w') as f:
        for i, col in enumerate(features_to_use):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('categorical: {}, numerical: {}, added: {}'.format(
        len(categorical_features), len(numeric_features), len(added_features)
    ))

    add_features(df)
    replace_categorical_features(df)

    df['log_price'] = np.log(df['price'] + 1.0)

    train_X = df[features_to_use].values
    train_y = df['log_price'].values

    # Run cv and produce out-of-fold predictions
    oof_preds = np.zeros(train_X.shape[0])
    # test_preds = np.zeros(test_X.shape[0]) # Can accumulate test preds/nfolds to get avg of fold models
    fold_gen = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=CV_SEED)
    for k, (dev_ix, val_ix) in enumerate(fold_gen.split(train_X)):
        dev_X, val_X = train_X[dev_ix, :], train_X[val_ix, :]
        dev_y, val_y = train_y[dev_ix], train_y[val_ix]

        if k == 0:  # First fold
            # Keep the number of rounds from first fold
            val_preds, model = run_xgb(params, n_rounds, dev_X, dev_y, val_X, val_y, EARLY_STOPPING)
            n_best = model.best_iteration
            logging.info('best iteration={}'.format(n_best))

            # Get feature importances
            importance = model.get_fscore(FMAP_FILE)
            imp_df = pd.DataFrame.from_dict(importance, 'index')
            imp_df.index.name = 'feature'
            imp_df.columns = ['fscore']
            imp_df.ix[:, 'fscore'] = imp_df.fscore / imp_df.fscore.sum()
            imp_df.sort_values('fscore', axis=0, ascending=False, inplace=True)
            imp_df.to_csv(FEAT_IMP_FILE, index=True)
            logging.info('feature importance is saved in {}'.format(FEAT_IMP_FILE))
        else:
            val_preds, model = run_xgb(params, n_best, dev_X, dev_y, val_X, val_y)

        oof_preds[val_ix] = val_preds  # save oof predictions per shuffled indices
        logging.info('CV #{}: {:.4f}'.format(k, mean_squared_error(val_y, val_preds)))

    logging.info('{}: CV {:.4f}, n_best {}'.format(feat_name, mean_squared_error(train_y, oof_preds), n_best))
    logging.info('Log file: %s' % log_file)


def run_xgb(params, n_rounds, train_X, train_y, test_X, test_y, early_stopping=None):

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [(xgtrain, 'train')]

    xgtest = xgb.DMatrix(test_X, label=test_y)
    watchlist.append((xgtest, 'val test'))
    model = xgb.train(params, xgtrain, n_rounds, watchlist, early_stopping_rounds=early_stopping)

    # TODO: Limit num trees to n_best rounds?
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="path of training data file")

    parser.add_argument("--depth", required=True, type=int, help="max tree depth")
    parser.add_argument("--lrate", required=True, type=float, help="learning rate eta")
    parser.add_argument("--n-rounds", required=True, type=int, help="max num training rounds")
    args = parser.parse_args()

    start = time.time()
    run(args.train_file, args.depth, args.lrate, args.n_rounds)

    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))


if __name__ == "__main__":
    main()
