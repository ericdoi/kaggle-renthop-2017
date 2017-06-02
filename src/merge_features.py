"""
Merge two sps features
"""

import pandas as pd
from kaggler.data_io import load_data
from scipy import sparse
from sklearn import datasets

root = '/Users/ericdoi/data-science/competitions/kaggle-two-sigma-renthop-2017/build/feature/'

# feature_name = 'fpq1'
# base_name = 'f1'
# pq_name = 'pq_500_3_0.1'

# feature_name = 'npq1'
# base_name = 'n1'
# pq_name = 'pq_500_3_0.1'

# feature_name = 'fpq2'
# base_name = 'f1'
# pq_name = 'pq_1000_3_0.1_int'

# feature_name = 'fpq3'
# base_name = 'f1'
# pq_name = 'pq_100_2_0.1_int'

# feature_name = 'fpq4'
# base_name = 'f1'
# pq_name = 'pq_130_12_0.2_int'

# feature_name = 'npq4'
# base_name = 'n1'
# pq_name = 'pq_130_12_0.2_ohe'

# feature_name = 'fpq5'
# base_name = 'f2'
# pq_name = 'pq_130_12_0.2_int'

# feature_name = 'f3pm1'
# base_name = 'f3'
# px_name = 'pm_1000_4_0.3'

# feature_name = 'f4pm1'
# base_name = 'f4'
# px_name = 'pm_1000_4_0.3'

# feature_name = 'n6pm2'
# base_name = 'n6'
# px_name = 'pm2_1000_4_0.3'

# feature_name = 'n6pq4'
# base_name = 'n6'
# px_name = 'pq_130_12_0.2_ohe'

# feature_name = 'f6img1'
# base_name = 'f6'
# px_name = 'img1'

# feature_name = 'f6aimg1'
# base_name = 'f6a'
# px_name = 'img1'

feature_name = 'n6aimg1'
base_name = 'n6a'
px_name = 'img1'

train_files = [
    root + '%s.trn.sps' % base_name,
    root + '%s.trn.sps' % px_name
]
test_files = [
    root + '%s.tst.sps' % base_name,
    root + '%s.tst.sps' % px_name
]
fmap_files = [
    root + '%s.fmap' % base_name,
    root + '%s.fmap' % px_name
]

train_out_path = root + '%s.trn.sps' % feature_name
test_out_path = root + '%s.tst.sps' % feature_name
fmap_out_path = root + '%s.fmap' % feature_name


def merge_features(files, out_path):
    combined_X = None
    first_y = None

    for ix, f in enumerate(files):
        [X, y] = load_data(f)
        if ix == 0:
            combined_X, first_y = X, y
        else:
            combined_X = sparse.hstack([combined_X, X])

    datasets.dump_svmlight_file(combined_X, first_y, out_path)


def main():
    merge_features(train_files, train_out_path)
    merge_features(test_files, test_out_path)

    # Merge fmap
    feat_df = pd.DataFrame()
    i = 0
    for fmap in fmap_files:
        f_df = pd.read_csv(fmap, delimiter='\t', header=None, names=['ix', 'name', 'q'])
        f_df['ix'] += i
        feat_df = pd.concat([feat_df, f_df])
        i += f_df.shape[0]
    feat_df.to_csv(fmap_out_path, sep='\t', index=False, header=None)


if __name__ == '__main__':
    main()
