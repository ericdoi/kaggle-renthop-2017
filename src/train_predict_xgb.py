"""
Thanks to:
https://github.com/jeongyoonlee/kaggler-template
https://www.kaggle.com/sudalairajkumar/xgb-starter-in-python
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from kaggler.data_io import load_data
from sklearn import model_selection
from sklearn.metrics import log_loss

DEFAULT_DEPTH = 6
DEFAULT_LRATE = 0.3
DEFAULT_NROUNDS = 200
CV_SEED = 2017
XG_SEED = 0
DEFAULT_FOLDS = 5
EARLY_STOPPING = 100


def initialize_logger(filename):
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=filename)

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.addHandler(handler)


def run(train_feature_file, test_feature_file, feature_map_file,
        predict_valid_file, predict_test_file, feature_importance_file,
        depth, lrate, n_rounds, n_folds=DEFAULT_FOLDS):

    model_name = os.path.splitext(
        os.path.splitext(os.path.basename(predict_test_file))[0]
    )[0]

    log_file = '{}.log'.format(model_name)
    initialize_logger(log_file)

    param_dict = {
        'objective': 'multi:softprob',
        'eta': lrate,
        'max_depth': depth,
        'silent': 1,
        'num_class': 3,
        'eval_metric': "mlogloss",
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': XG_SEED
    }

    params = list(param_dict.items())

    [train_X, train_y] = load_data(train_feature_file) #datasets.load_svmlight_file(train_feature_file)
    [test_X, _] = load_data(test_feature_file) #datasets.load_svmlight_file(test_feature_file)

    # Run cv and produce out-of-fold predictions
    oof_preds = np.zeros((train_X.shape[0], 3))
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
            importance = model.get_fscore(feature_map_file)
            imp_df = pd.DataFrame.from_dict(importance, 'index')
            imp_df.index.name = 'feature'
            imp_df.columns = ['fscore']
            imp_df.ix[:, 'fscore'] = imp_df.fscore / imp_df.fscore.sum()
            imp_df.sort_values('fscore', axis=0, ascending=False, inplace=True)
            imp_df.to_csv(feature_importance_file, index=True)
            logging.info('feature importance is saved in {}'.format(feature_importance_file))
        else:
            val_preds, model = run_xgb(params, n_best, dev_X, dev_y, val_X, val_y)

        oof_preds[val_ix] = val_preds  # save oof predictions per shuffled indices

        logging.info('CV #{}: {:.4f}'.format(k, log_loss(val_y, val_preds)))

    logging.info('Saving validation predictions...')
    oof_preds_df = pd.DataFrame(oof_preds)
    oof_preds_df.columns = ['low', 'medium', 'high']
    oof_preds_df.to_csv(predict_valid_file, index=False)

    # Run on 100% training
    logging.info('Retraining with 100% training data')
    test_preds, model = run_xgb(params, n_best, train_X, train_y, test_X)
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df.columns = ['low', 'medium', 'high']

    logging.info('Saving test predictions...')
    test_preds_df.to_csv(predict_test_file, index=False)

    logging.info('{}: CV {:.4f}, n_best {}'.format(model_name, log_loss(train_y, oof_preds), n_best))
    logging.info('Log file: %s' % log_file)


def run_xgb(params, n_rounds, train_X, train_y, test_X, test_y=None, early_stopping=None):

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [(xgtrain, 'train')]

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist.append((xgtest, 'val test'))
        model = xgb.train(params, xgtrain, n_rounds, watchlist, early_stopping_rounds=early_stopping)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(params, xgtrain, n_rounds, watchlist)

    # TODO: Limit num trees to n_best rounds?
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feature-file", required=True, help="input path for train features")
    parser.add_argument("--test-feature-file", required=True, help="input path for test features")
    parser.add_argument('--feature-map-file', required=True, help="input path for feature indices")

    parser.add_argument("--predict-valid-file", required=True, help="output path for oof validation preds")
    parser.add_argument("--predict-test-file", required=True, help="output path for test predictions")
    parser.add_argument('--feature-importance-file', required=True, help="output path for importances")

    parser.add_argument("--depth", default=DEFAULT_DEPTH, type=int, help="max tree depth")
    parser.add_argument("--lrate", default=DEFAULT_LRATE, type=float, help="learning rate eta")
    parser.add_argument("--n-rounds", default=DEFAULT_NROUNDS, type=int, help="max num training rounds")
    args = parser.parse_args()

    start = time.time()

    run(args.train_feature_file, args.test_feature_file, args.feature_map_file,
        args.predict_valid_file, args.predict_test_file, args.feature_importance_file,
        args.depth, args.lrate, args.n_rounds)

    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) / 60))


if __name__ == "__main__":
    main()
