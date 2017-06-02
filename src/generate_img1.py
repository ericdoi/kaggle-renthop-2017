import argparse
import logging
import os
import pdb
import time

import numpy as np
import pandas as pd
from kaggler.preprocessing import OneHotEncoder, Normalizer
from scipy import sparse
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer


numeric_features = [
    'avg_filesize',
    'avg_filesize_top2',
    'avg_histogram_skew',
    'avg_histogram_skew_top2',
    'avg_hw_ratio',
    'avg_hw_ratio_top2',
    'avg_img_i',
    'avg_img_i_top2',
    'avg_img_q',
    'avg_img_q_top2',
    'avg_img_y',
    'avg_img_y_top2',
    'avg_sharpness_90pct',
    'ct_portrait',
    'ct_portrait_top2',
    'has_floorplan',
    'max_filesize',
    'max_histogram_skew',
    'max_hw_ratio',
    'max_img_i',
    'max_img_q',
    'max_img_y',
    'max_num_exif_tags',
    'max_sharpness_90pct',
    'min_filesize',
    'min_histogram_skew',
    'min_hw_ratio',
    'min_img_i',
    'min_img_q',
    'min_img_y',
    'min_sharpness_90pct',
    'sum_ocr_len',
    'sum_ocr_len_top2',

    # added
    'null_img_feats',
    'yiq_cluster_uniqueness'
]

categorical_features_to_convert = ['yiq_clusters', 'yiq_clusters_top2']

target_num_map = {'low': 0, 'medium': 1, 'high': 2}


def add_features(df):
    # nan/nulls are all-or-none, so just add a dummy var for null for linear feats
    df['null_img_feats'] = df.isnull().any(axis=1) * 1

    # Add cluster uniqueness
    df['yiq_cluster_uniqueness'] = df['yiq_clusters'].apply(
        lambda cs: len(set(cs)) * 1.0 / len(cs) if type(cs) == list else 0
    )


def load(train_path, test_path):
    with open(train_path) as train_file:
        with open(test_path) as test_file:
            train_df = pd.read_json(train_file)
            test_df = pd.read_json(test_file)
            print 'Train data:', train_df.shape
            print 'Test data:', test_df.shape
            return train_df, test_df


def load_and_concat_agg_feats(aggs_path):
    agg_dfs = []
    for agg_name in os.listdir(aggs_path):
        agg_path = os.path.join(aggs_path, agg_name)
        agg_df = pd.read_json(agg_path, orient='split').set_index('listing_id')
        agg_dfs.append(agg_df)
        print 'agg file: %s (%s)' % (agg_name, agg_df.shape[0])
    img_df = pd.concat(agg_dfs, axis=0)

    return img_df


def make_list_features(df):
    """Combined df version"""
    print(df["features"].head())
    ctr = CountVectorizer()
    X_text = ctr.fit_transform(df["features"])

    tag_names = ['tag_%s' % tag for tag in ctr.get_feature_names()]
    return X_text, tag_names


def normalize_features(df, cols):
    nm = Normalizer()
    df.ix[:, cols] = nm.fit_transform(df[cols].values)


def initialize_logger(filename):
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=filename)

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.addHandler(handler)


def run(aggs_path, train_file, test_file, train_feature_file, test_feature_file, feature_map_file):

    feat_name = os.path.splitext(
        os.path.splitext(os.path.basename(test_feature_file))[0]
    )[0]

    log_file = '{}.log'.format(feat_name)
    initialize_logger(log_file)

    logging.info('loading raw data')

    train_df, test_df = load(train_file, test_file)
    n_trn_rows = train_df.shape[0]
    n_tst_rows = test_df.shape[0]

    img_df = load_and_concat_agg_feats(aggs_path)

    # Reindex to preserve the data order.  If entries are missing, let them be NaNs
    train_img_df = img_df.reindex(train_df['listing_id'])
    test_img_df = img_df.reindex(test_df['listing_id'])
    print train_img_df.shape
    print test_img_df.shape

    add_features(train_img_df)
    add_features(test_img_df)

    full_df = pd.concat([train_img_df, test_img_df], axis=0)

    # Clean up features
    full_df[numeric_features] = full_df[numeric_features].fillna(0)

    normalize_features(full_df, numeric_features)

    # Treat cluster ids like text features
    for col in categorical_features_to_convert:
        full_df[col] = full_df[col].apply(
            lambda lst: " ".join(['%s_%s' % (col, k) for k in lst]) if type(lst) == list else ''
        )
    full_df['features'] = full_df[categorical_features_to_convert].apply(lambda row: ' '.join(row), axis=1)

    X_text, tag_names = make_list_features(full_df)

    full_X = sparse.hstack([full_df[numeric_features], X_text]).tocsr()

    # Save feature mapping
    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(numeric_features + tag_names):
            f.write('{}\t{}\tq\n'.format(i, col))

    train_y = np.array(
        train_df['interest_level'].apply(lambda x: target_num_map[x])
    )

    test_y = np.zeros(n_tst_rows)

    logging.info('saving features')
    datasets.dump_svmlight_file(full_X[:n_trn_rows, ], train_y, train_feature_file)
    datasets.dump_svmlight_file(full_X[n_trn_rows:, ], test_y, test_feature_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggs-path", required=True, help="path of dir with agg image feature jsons")
    parser.add_argument("--train-file", required=True, help="path of training data file")
    parser.add_argument("--test-file", required=True, help="path of test data file")
    parser.add_argument("--train-feature-file", required=True, help="output path for train features")
    parser.add_argument("--test-feature-file", required=True, help="output path for test features")
    parser.add_argument('--feature-map-file', required=True, help="output path for feature indices")

    args = parser.parse_args()

    start = time.time()
    run(args.aggs_path,
        args.train_file, args.test_file,
        args.train_feature_file, args.test_feature_file,
        args.feature_map_file)

    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))


if __name__ == "__main__":
    main()
