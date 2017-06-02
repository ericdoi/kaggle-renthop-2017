"""
Add more CV features (range), building
"""
import argparse
import logging
import pdb
import time
from collections import Counter

import numpy as np
import pandas as pd
from kaggler.preprocessing import LabelEncoder
from scipy import sparse
from sklearn import datasets
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

N_FOLDS_CV_FEATS = 5
CV_SEED = 2017

LONG_FEAT_CHARS = 27

numeric_features = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

added_features = ["num_photos", "num_features", "has_long_feature",
                  "has_all_caps_feature", "num_description_words", "created_year",
                  "created_month", "created_day", "listing_id", "created_hour",
                  "created_dow", "has_st_num", "valid_addr_suffix",
                  "price_per_room", "price_per_bed", "price_per_bath", "beds_minus_baths",
                  "rooms", "bed_ratio"]

categorical_features = ["display_address", "manager_id", "building_id", "street_address", "disp_addr_suffix"]

st_abbreviations = {
    'street': 'st',
    'avenue': 'ave',
    'place': 'pl',
    'boulevard': 'blvd',
    'lane': 'ln',
    'road': 'rd',
    'parkway': 'pkwy',
    'court': 'ct',
    'plaza': 'plz',
    'terrace': 'ter',
    'center': 'ctr',
    'square': 'sq'
}

valid_suffixes = set(st_abbreviations.values())


def has_street_num(address_str):
    return 1 if len(address_str) > 0 and address_str.strip()[0].isdigit() else 0


def get_address_suffix(address_str):
    parts = address_str.split()
    if len(parts) == 0:
        return ''
    else:
        suffix = parts[-1].lower().strip('.,')
        return st_abbreviations.get(suffix, suffix)


def is_standard_address_suffix(address_str):
    return 1 if address_str in valid_suffixes else 0


def add_features(df):
    # count of photos #
    df["num_photos"] = df["photos"].apply(len)

    # count of "features" #
    df["num_features"] = df["features"].apply(len)
    df["has_long_feature"] = df["features"].apply(
        lambda lst: 1 * (max((len(tag) for tag in lst + [""])) >= LONG_FEAT_CHARS)
    )
    df["has_all_caps_feature"] = df["features"].apply(
        lambda lst: 1 * any([len(tag) >= 5 and tag == tag.upper() for tag in lst])
    )

    # count of words present in description column #
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))

    # convert the created column to datetime object so as to extract more features
    df["created"] = pd.to_datetime(df["created"])

    # Let us extract some features like year, month, day, hour from date columns #
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    df['created_dow'] = df["created"].dt.dayofweek

    df['has_st_num'] = df['street_address'].apply(has_street_num)
    df['disp_addr_suffix'] = df['display_address'].apply(get_address_suffix)
    df['valid_addr_suffix'] = df['disp_addr_suffix'].apply(is_standard_address_suffix)

    df['rooms'] = df['bedrooms'] + df['bathrooms']
    df['beds_minus_baths'] = df['bedrooms'] - df['bathrooms']
    df['price_per_room'] = (df['price'] / df['rooms']).replace(np.inf, 10**6)
    df['price_per_bed'] = (df['price'] / df['bedrooms']).replace(np.inf, 10**6)
    df['price_per_bath'] = (df['price'] / df['bathrooms']).replace(np.inf, 10**6)
    df['bed_ratio'] = (df['bedrooms'] / df['rooms']).replace(np.inf, 10**6)


def replace_categorical_features(train_df, test_df, cat_cols, min_obs=10):
    # Would have to save label encoders for each categorical feature to split train/test processing.
    # if train_df[f].dtype == 'object':
    lbl = LabelEncoder(min_obs)
    cat_df = pd.concat([train_df, test_df], axis=0)[cat_cols]
    cat_df.ix[:, :] = lbl.fit_transform(cat_df.values)

    n_trn, n_tst = (train_df.shape[0], test_df.shape[0])
    train_df.ix[:, cat_cols] = cat_df.values[:n_trn, ]
    test_df.ix[:, cat_cols] = cat_df.values[n_trn:, ]


# TODO: Inefficient since LabelEncoder already computes counts
def add_small_count_features(train_df, test_df, cat_cols, max_obs=10**5):
    cat_df = pd.concat([train_df, test_df], axis=0)[cat_cols]

    count_feature_names = []
    for col in cat_cols:
        feat_ct_name = '%s_ct' % col
        count_feature_names.append(feat_ct_name)
        ctr = Counter(cat_df[col])
        train_df[feat_ct_name] = train_df[col].apply(lambda x: min(ctr[x], max_obs))
        test_df[feat_ct_name] = test_df[col].apply(lambda x: min(ctr[x], max_obs))

    print 'Small count features:'
    print train_df[count_feature_names].head()
    return count_feature_names


def add_cv_features(train_df, test_df, gen_fxn, n_folds=N_FOLDS_CV_FEATS):
    """
    Generate ofeatures based off of out-of-fold aggregations across manager_id and the target
    Thanks to gdy5:  https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31107
    """
    print "Shapes before cv features: Train %s | Test %s" % (train_df.shape, test_df.shape)

    # To generate features for current "validation" fold, use aggregation on other k-1 "development" folds.
    fold_gen = model_selection.KFold(n_splits=n_folds, shuffle=True) #, random_state=CV_SEED)
    aggs_df_list = []  # Accumulate the fold-specific feature dfs
    for k, (dev_ix, val_ix) in enumerate(fold_gen.split(train_df.values)):
        dev_df = train_df.iloc[dev_ix]
        val_df = train_df.iloc[val_ix]

        fold_aggs_df, _ = gen_fxn(dev_df, val_df)
        aggs_df_list.append(fold_aggs_df)

    # Vertically stack the fold rows
    train_aggs_df = pd.concat(aggs_df_list, axis=0)
    test_aggs_df, added_cv_feats = gen_fxn(train_df, test_df)

    # Horizontally stack the new feats. Use join_axes to preserve index order.
    train_df_2 = pd.concat([train_df, train_aggs_df], axis=1, join='inner', join_axes=[train_df.index])
    test_df_2 = pd.concat([test_df, test_aggs_df], axis=1, join='inner', join_axes=[test_df.index])

    print 'CV features:'
    print test_aggs_df.head()
    print "Shapes after cv features: Train %s | Test %s" % (train_df_2.shape, test_df_2.shape)
    return train_df_2, test_df_2, added_cv_feats


def compute_manager_cv_features(dev_df, val_df):
    """
    Use dev_df to compute manager stats, then use them to annotate val_df
    """
    grouped_df = dev_df.groupby(['manager_id', 'interest_level'])

    # Start with listing id counts
    df = grouped_df['listing_id'].count().unstack()
    # Compute manager skill, thanks to den3b81:
    # https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/improve-perfomances-using-manager-features
    df['manager_quality'] = (df['high'] * 2 + df['medium']) / (df.sum(axis=1))
    aggs_df = df.rename(columns={
        'low': 'manager_low_count',
        'medium': 'manager_medium_count',
        'high': 'manager_high_count'})

    # Add more complicated aggregations
    interest_aggs_mean = grouped_df.agg({
        'price': 'mean',
        'price_per_room': 'mean',
        'price_per_bed': 'mean',
        'price_per_bath': 'mean',
        'bedrooms': 'mean',
        'bathrooms': 'mean',
        'rooms': 'mean'
    }).unstack()
    interest_aggs_range = grouped_df.agg({
        'price': lambda x: np.max(x) - np.min(x),
        'price_per_room': lambda x: np.max(x) - np.min(x)
    }).unstack()

    added_cv_feats = ['manager_quality', 'manager_low_count', 'manager_medium_count', 'manager_high_count']

    # Rename aggregation features
    for (fxn, agg_type_df) in [('mean', interest_aggs_mean),
                               ('range', interest_aggs_range)]:
        for agg_feat in ['bedrooms', 'bathrooms', 'rooms', 'price', 'price_per_room', 'price_per_bed',
                         'price_per_bath']:
            if agg_feat not in agg_type_df.columns:
                continue
            df = agg_type_df[agg_feat]
            for level in ['low', 'medium', 'high']:
                fname = 'manager_%s_%s_%s' % (level[0], fxn, agg_feat)
                aggs_df[fname] = df[level]
                added_cv_feats.append(fname)

    # Join with train_df indices
    aggs_df = pd.merge(
        val_df[['manager_id']],
        aggs_df,
        left_on='manager_id',
        right_index=True
    )
    del aggs_df['manager_id']

    return aggs_df, added_cv_feats


def compute_building_cv_features(dev_df, val_df):
    """
    Use dev_df to compute manager stats, then use them to annotate val_df
    """
    grouped_df = dev_df.groupby(['building_id', 'interest_level'])

    # Start with listing id counts
    df = grouped_df['listing_id'].count().unstack()
    # Compute manager skill, thanks to den3b81:
    # https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/improve-perfomances-using-manager-features
    df['bldg_quality'] = (df['high'] * 2 + df['medium']) / (df.sum(axis=1))
    aggs_df = df.rename(columns={
        'low': 'bldg_low_count',
        'medium': 'bldg_medium_count',
        'high': 'bldg_high_count'})

    # Add more complicated aggregations
    interest_aggs_mean = grouped_df.agg({
        'price': 'mean',
        'price_per_room': 'mean',
        'price_per_bed': 'mean',
        'price_per_bath': 'mean',
        'bedrooms': 'mean',
        'bathrooms': 'mean',
        'rooms': 'mean'
    }).unstack()
    interest_aggs_range = grouped_df.agg({
        'price': lambda x: np.max(x) - np.min(x),
        'price_per_room': lambda x: np.max(x) - np.min(x)
    }).unstack()

    added_cv_feats = ['bldg_quality', 'bldg_low_count', 'bldg_medium_count', 'bldg_high_count']

    # Rename aggregation features
    for (fxn, agg_type_df) in [('mean', interest_aggs_mean),
                               ('range', interest_aggs_range)]:
        for agg_feat in ['bedrooms', 'bathrooms', 'rooms', 'price', 'price_per_room', 'price_per_bed',
                         'price_per_bath']:
            if agg_feat not in agg_type_df.columns:
                continue
            df = agg_type_df[agg_feat]
            for level in ['low', 'medium', 'high']:
                fname = 'bldg_%s_%s_%s' % (level[0], fxn, agg_feat)
                aggs_df[fname] = df[level]
                added_cv_feats.append(fname)

    # Join with train_df indices
    aggs_df = pd.merge(
        val_df[['building_id']],
        aggs_df,
        left_on='building_id',
        right_index=True
    )
    del aggs_df['building_id']

    return aggs_df, added_cv_feats


def make_text_features(train_df, test_df):
    train_df['features'] = train_df["features"].apply(
        lambda lst: " ".join(["_".join(tag.split()) for tag in lst])
    )
    test_df['features'] = test_df["features"].apply(
        lambda lst: " ".join(["_".join(tag.split()) for tag in lst])
    )
    print(train_df["features"].head())
    ctr = CountVectorizer(stop_words='english', max_features=200)
    trn_sparse_text = ctr.fit_transform(train_df["features"])
    tst_sparse_text = ctr.transform(test_df["features"])

    tag_names = ['tag_%s' % tag for tag in ctr.get_feature_names()]
    return trn_sparse_text, tst_sparse_text, tag_names


def load(train_path, test_path):
    with open(train_path) as train_file:
        with open(test_path) as test_file:
            train_df = pd.read_json(train_file)
            test_df = pd.read_json(test_file)
            print 'Train data:', train_df.shape
            print 'Test data:', test_df.shape
            return train_df, test_df


def run(train_file, test_file, train_feature_file, test_feature_file, feature_map_file,
        test_id_file, train_y_file, test_y_file):

    logging.info('loading raw data')

    train_df, test_df = load(train_file, test_file)

    # Make sure index order is maintained between val_df and aggs_df
    before_test_listing_id_order = test_df['listing_id'].head().values

    if test_id_file is not None:
        test_df['listing_id'].to_csv(test_id_file, index=False, header=True)

    logging.info('categorical: {}, numerical: {}, added: {}'.format(
        len(categorical_features), len(numeric_features), len(added_features)
    ))

    add_features(train_df)
    add_features(test_df)
    train_df, test_df, added_manager_cv_feats = add_cv_features(train_df, test_df, compute_manager_cv_features)
    train_df, test_df, added_building_cv_feats = add_cv_features(train_df, test_df, compute_building_cv_features)
    count_feature_names = add_small_count_features(train_df, test_df, categorical_features)
    replace_categorical_features(train_df, test_df, categorical_features)

    print 'Make sure test data order was maintained:'
    after_test_listing_id_order = test_df['listing_id'].head().values
    assert ((before_test_listing_id_order == after_test_listing_id_order).all())
    print 'Before:', before_test_listing_id_order
    print 'After:', after_test_listing_id_order

    features_to_use = numeric_features + added_manager_cv_feats + added_building_cv_feats + \
        added_features + categorical_features + count_feature_names

    trn_sparse_text, tst_sparse_text, tag_names = make_text_features(train_df, test_df)

    # Treat nan as 0 so they are treated as missing in sparse representation.
    # Tianqi says usually don't need to worry about 0s conflated with NaNs (https://github.com/dmlc/xgboost/issues/21)
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    train_X = sparse.hstack([train_df[features_to_use], trn_sparse_text]).tocsr()
    test_X = sparse.hstack([test_df[features_to_use], tst_sparse_text]).tocsr()

    # Save feature mapping
    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(features_to_use + tag_names):
            f.write('{}\t{}\tq\n'.format(i, col))

    target_num_map = {'low': 0, 'medium': 1, 'high': 2}
    train_y = np.array(
        train_df['interest_level'].apply(lambda x: target_num_map[x])
    )

    test_y = np.zeros(test_X.shape[0])

    # Save labels
    pd.Series(train_y, name='interest_level').to_csv(train_y_file, index=False, header=True)
    pd.Series(test_y, name='interest_level').to_csv(test_y_file, index=False, header=True)

    logging.info('saving features')
    datasets.dump_svmlight_file(train_X, train_y, train_feature_file)
    datasets.dump_svmlight_file(test_X, test_y, test_feature_file)


def main():
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="path of training data file")
    parser.add_argument("--test-file", required=True, help="path of test data file")
    parser.add_argument("--train-feature-file", required=True, help="output path for train features")
    parser.add_argument("--test-feature-file", required=True, help="output path for test features")
    parser.add_argument('--feature-map-file', required=True, help="output path for feature indices")
    parser.add_argument("--test-id-file", required=False, help="output path for ids (with header)")
    parser.add_argument("--train-y-file", required=False, help="output path for target in train data")
    parser.add_argument("--test-y-file", required=False, help="output path for target in test data")
    args = parser.parse_args()

    start = time.time()
    run(args.train_file, args.test_file,
        args.train_feature_file, args.test_feature_file,
        args.feature_map_file, args.test_id_file,
        args.train_y_file, args.test_y_file)

    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))


if __name__ == "__main__":
    main()
