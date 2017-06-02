"""
Compile all the image-level intermediate feature files to get the average color per image (y,i,q space)
Run k-means to generate model
Use model to create aggregated listing-level image features

cat ../build/photo-features/tmp/* | cut -d, -f5,6,7 | head -3
cat ../build/photo-features/tmp/* | cut -d, -f5,6,7 | grep -v img > ../build/photo-features/img_yiq_allsample.csv
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


train_frac = 1.0
n_clusters = 10
yiq_file = '../build/photo-features/img_yiq_allsample.csv'
kmeans_out_file = '../build/photo-features/trained_kmeans_%s_allsample.pkl' % n_clusters


def main():
    all_yiq_df = pd.read_csv(yiq_file, header=None, names=['y', 'i', 'q'])
    sm_yiq_df = all_yiq_df.sample(frac=train_frac)

    kmeans = Pipeline([
        ('standardize', StandardScaler()),
        ('cluster', KMeans(n_clusters, random_state=0))
    ])
    kmeans.fit(sm_yiq_df.values)
    joblib.dump(kmeans, kmeans_out_file)


if __name__ == '__main__':
    main()
