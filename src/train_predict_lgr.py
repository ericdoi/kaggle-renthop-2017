"""
Thanks to:
https://github.com/jeongyoonlee/kaggler-template
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from kaggler.data_io import load_data
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

DEFAULT_SOLVER = 'sag'
DEFAULT_MULTICLASS = 'multinomial'
DEFAULT_C = 0.25

CV_SEED = 2017
DEFAULT_FOLDS = 5


def initialize_logger(filename):
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=filename)

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.addHandler(handler)


def run(train_feature_file, test_feature_file, feature_map_file,
        predict_valid_file, predict_test_file, feature_importance_file,
        solver, multiclass, C, n_folds=DEFAULT_FOLDS):

    model_name = os.path.splitext(
        os.path.splitext(os.path.basename(predict_test_file))[0]
    )[0]

    log_file = '{}.log'.format(model_name)
    initialize_logger(log_file)

    param_dict = {
        'solver': solver,
        'multi_class': multiclass,
        'max_iter': 1000,
        'tol': 0.0001,
        'C': C
    }

    [train_X, train_y] = load_data(train_feature_file)  # datasets.load_svmlight_file(train_feature_file)
    [test_X, _] = load_data(test_feature_file)  # datasets.load_svmlight_file(test_feature_file)

    # Run cv and produce out-of-fold predictions
    oof_preds = np.zeros((train_X.shape[0], 3))
    # test_preds = np.zeros(test_X.shape[0]) # Can accumulate test preds/nfolds to get avg of fold models
    fold_gen = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=CV_SEED)
    for k, (dev_ix, val_ix) in enumerate(fold_gen.split(train_X)):
        dev_X, val_X = train_X[dev_ix, :], train_X[val_ix, :]
        dev_y, val_y = train_y[dev_ix], train_y[val_ix]

        val_preds, model = run_lr(param_dict, dev_X, dev_y, val_X, val_y)

        oof_preds[val_ix] = val_preds  # save oof predictions per shuffled indices

        logging.info('CV #{}: {:.4f}'.format(k, log_loss(val_y, val_preds)))

    logging.info('CV: {:.4f}'.format(log_loss(train_y, oof_preds)))
    logging.info('Saving validation predictions...')
    oof_preds_df = pd.DataFrame(oof_preds)
    oof_preds_df.columns = ['low', 'medium', 'high']
    oof_preds_df.to_csv(predict_valid_file, index=False)

    # Run on 100% training
    logging.info('Retraining with 100% training data')
    test_preds, model = run_lr(param_dict, train_X, train_y, test_X)

    # Save feature importances
    save_feature_importance(model, feature_map_file, feature_importance_file)

    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df.columns = ['low', 'medium', 'high']

    logging.info('Saving test predictions...')
    test_preds_df.to_csv(predict_test_file, index=False)

    logging.info('{}: CV {:.4f}'.format(model_name, log_loss(train_y, oof_preds)))
    logging.info('Log file: %s' % log_file)


def save_feature_importance(model, feature_map_file, feature_importance_file):
    mean_wts = np.mean(np.abs(model.coef_), axis=0)

    names = []
    for line in open(feature_map_file):
        ix, name, _ = line.split('\t')
        names.append(name)

    feat_imp = zip(names, mean_wts)
    feat_df = pd.DataFrame(feat_imp, columns=['feature', 'score'])
    feat_df = feat_df.sort_values(by='score', ascending=False)
    feat_df.to_csv(feature_importance_file, index=False)
    logging.info('feature importance is saved in {}'.format(feature_importance_file))


def run_lr(params, train_X, train_y, test_X, test_y=None):
    model = LogisticRegression(random_state=42, warm_start=True, max_iter=300)
    model.set_params(**params)
    model.fit(train_X, train_y)

    pred_trn_y = model.predict_proba(train_X)
    pred_tst_y = model.predict_proba(test_X)

    trn_loss = log_loss(train_y, pred_trn_y)
    logging.info('train: {:.4f}'.format(trn_loss))

    return pred_tst_y, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feature-file", required=True, help="input path for train features")
    parser.add_argument("--test-feature-file", required=True, help="input path for test features")
    parser.add_argument('--feature-map-file', required=True, help="input path for feature indices")

    parser.add_argument("--predict-valid-file", required=True, help="output path for oof validation preds")
    parser.add_argument("--predict-test-file", required=True, help="output path for test predictions")
    parser.add_argument('--feature-importance-file', required=True, help="output path for importances")

    parser.add_argument("--solver", default=DEFAULT_SOLVER, type=str, help="sag, newton-cg, liblinear, lbfgs")
    parser.add_argument("--multiclass", required=DEFAULT_MULTICLASS, type=str, help="multinomial, ovr")
    parser.add_argument("--C", required=DEFAULT_C, type=float, help="regularization")
    args = parser.parse_args()

    start = time.time()

    run(args.train_feature_file, args.test_feature_file, args.feature_map_file,
        args.predict_valid_file, args.predict_test_file, args.feature_importance_file,
        args.solver, args.multiclass, args.C)

    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) / 60))


if __name__ == "__main__":
    main()
