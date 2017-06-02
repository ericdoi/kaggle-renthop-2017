"""
Generate image features with multiple processes
1) Generate a file for each listing with one feature row per image (or load if it already exists)
2) Aggregate all image features into a single feature per listing

"""
import argparse
import colorsys
import logging
import multiprocessing
import os
import shutil
import stat
import time

import numpy as np
import pandas as pd
import pytesseract
import scipy
import scipy.ndimage
from PIL import Image
from skimage import io, color, util
from sklearn.externals import joblib


# Generated in Feature Exploration notebook.  Map from listing_id to ordered photo list.
def load_photo_ordering(order_file):
    load_series = pd.read_csv(order_file, index_col='listing_id')['photos'].fillna('')
    test_dict = load_series.apply(lambda joined: joined.split(';')).to_dict()
    return test_dict


photo_file_ordering_dict = load_photo_ordering('../build/photo-features/listing_to_photo_file_order.csv')

kmeans_loaded = joblib.load('../build/photo-features/trained_kmeans_10_allsample.pkl')

OCR_DEL_CHARS = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
IMG_FEATS = ['num_exif_tags', 'hw_ratio', 'portrait', 'img_y', 'img_i', 'img_q', 'filesize',
             'histogram_skew', 'ocr_words', 'is_floorplan', 'sharpness_90pct']
FLOOR_PLAN_CUTOFF = 0.4


def get_histogram_skew(img):
    hist, bins = np.histogram(img, bins=50)
    coverage_of_top_bins = np.sort(hist)[-4:].sum()*1.0 / hist.sum()
    return coverage_of_top_bins


def num_exif_tags(im):
    info = im._getexif()
    if info is None:
        return 0
    else:
        return len(info)


def make_image_feats(img_file):
    pil_img = Image.open(img_file)
    img = io.imread(img_file)

    if pil_img.size[0] < 20 or pil_img.size[1] < 20:
        logging.error('Image too small: %s' % img_file)
        return None

    xcrop, ycrop = pil_img.size[0]/2, pil_img.size[1]/2

    filestat = os.stat(img_file)

    hw_ratio = img.shape[1] * 1.0 / img.shape[0]
    portrait = 0 if hw_ratio > 1 else 1
    img_rgb = img.mean(axis=(0,1))
    # img_hsv = colorsys.rgb_to_hsv(img_rgb[0], img_rgb[1], img_rgb[2])
    img_yiq = colorsys.rgb_to_yiq(img_rgb[0], img_rgb[1], img_rgb[2])
    histogram_skew = get_histogram_skew(img)

    # Sharpness
    img_grey = util.crop(color.rgb2grey(img), [(0, xcrop), (0, ycrop)])
    laplace = scipy.ndimage.laplace(img_grey)
    sharpness_90pct = np.percentile(np.abs(laplace.ravel()), 90)

    # OCR part of the image to save time
    pil_img_cropped = pil_img.crop((0, 0, xcrop, ycrop))
    ocr_words = pytesseract.image_to_string(pil_img_cropped).translate(None, OCR_DEL_CHARS).split()

    # Try "color signature" of 3 yiq vectors from clustering?  Not really any better.
    # # kmeans on small image rgb
    # sm_img_flat = sm_img.reshape((smx * smy, -1))
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(sm_img_flat)
    # color_signature = []
    # for (r, g, b) in kmeans.cluster_centers_:
    #     (y, i, q) = colorsys.rgb_to_yiq(r, g, b)
    #     color_signature.append([y, i])
    # color_signature.sort()
    # color_signature = np.array([x for yi in color_signature for x in yi]) # flatten

    dct = {
        'filepath': img_file.split('/')[-1],
        'num_exif_tags': num_exif_tags(pil_img),
        # 'hw': im.shape,
        'hw_ratio': max(hw_ratio, 1.0 / hw_ratio),
        'portrait': portrait,

        'img_y': img_yiq[0],
        'img_i': img_yiq[1],
        'img_q': img_yiq[2],
        'filesize': filestat[stat.ST_SIZE],
        'histogram_skew': histogram_skew,
        'ocr_words': ocr_words,
        'is_floorplan': 1.0 * (histogram_skew > FLOOR_PLAN_CUTOFF) * (len(ocr_words) > 0),
        'sharpness_90pct': sharpness_90pct
        # 'color_signature': color_signature
    }

    return dct


def get_image_feat_df(listing_id, img_files, out_file):
    """Load image feats. If it doesn't exist, create and save."""
    # Ensure that it contains photos and output is not already done
    if os.path.exists(out_file):
        df = pd.read_csv(out_file, index_col='filepath')
        df['ocr_words'] = df['ocr_words'].fillna('').apply(lambda joined: str(joined).split(';'))
        return df.fillna(0)
    else:
        feat_list = []
        for f in img_files:
            if f[-3:].lower() != 'jpg':
                continue
            try:
                dct = make_image_feats(f)
                if dct is not None:
                    feat_list.append(dct)
            except StandardError, e:
                logging.error('Make image on %s failed unexpectedly: %s' % (f, e))

        df = pd.DataFrame(feat_list)

        photo_ordering = photo_file_ordering_dict[listing_id]  # Get what order photos should be in
        if df.shape[0] == 0:
            return None
        df = df.set_index('filepath')
        dupes = df.index.duplicated()
        if dupes.sum() > 0:
            logging.error('Duplicate filepath indices: %s, %s' % (listing_id, df.head()))
            df = df[~dupes]
        df = df.reindex(photo_ordering).dropna()  # Drop missing filenames from photo_ordering
        df['ocr_words'] = df['ocr_words'].apply(lambda words: ';'.join(words))
        df.to_csv(out_file, columns=IMG_FEATS, index=True)
        return df


def aggregate_image_feats(df):
    agg_dict = dict()

    # Features for first 2 photos
    top2_df = df.head(2)
    agg_dict['avg_hw_ratio_top2'] = top2_df['hw_ratio'].mean()
    agg_dict['avg_img_y_top2'] = top2_df['img_y'].mean()
    agg_dict['avg_img_i_top2'] = top2_df['img_i'].mean()
    agg_dict['avg_img_q_top2'] = top2_df['img_q'].mean()

    agg_dict['std_img_y_top2'] = top2_df['img_y'].std()
    agg_dict['std_img_i_top2'] = top2_df['img_i'].std()
    agg_dict['std_img_q_top2'] = top2_df['img_q'].std()

    agg_dict['avg_filesize_top2'] = top2_df['filesize'].mean()
    agg_dict['avg_histogram_skew_top2'] = top2_df['histogram_skew'].mean()
    agg_dict['avg_portrait_top2'] = top2_df['portrait'].mean()
    agg_dict['sum_ocr_len_top2'] = top2_df['ocr_words'].apply(lambda x: len(x)).sum()

    # Features for all photos
    agg_dict['max_num_exif_tags'] = df['num_exif_tags'].max()
    agg_dict['avg_hw_ratio'] = df['hw_ratio'].mean()
    agg_dict['min_hw_ratio'] = df['hw_ratio'].min()
    agg_dict['max_hw_ratio'] = df['hw_ratio'].max()

    agg_dict['avg_img_y'] = df['img_y'].mean()
    agg_dict['avg_img_i'] = df['img_i'].mean()
    agg_dict['avg_img_q'] = df['img_q'].mean()
    agg_dict['std_img_y'] = df['img_y'].std()
    agg_dict['std_img_i'] = df['img_i'].std()
    agg_dict['std_img_q'] = df['img_q'].std()

    agg_dict['avg_filesize'] = df['filesize'].mean()
    agg_dict['max_filesize'] = df['filesize'].max()
    agg_dict['std_filesize'] = df['filesize'].std()

    agg_dict['avg_histogram_skew'] = df['histogram_skew'].mean()
    agg_dict['std_histogram_skew'] = df['histogram_skew'].std()

    agg_dict['sum_ocr_len'] = df['ocr_words'].apply(lambda x: len(x)).sum()
    agg_dict['has_floorplan'] = df['is_floorplan'].sum()
    agg_dict['avg_portrait'] = df['portrait'].mean()

    agg_dict['avg_sharpness_90pct'] = df['sharpness_90pct'].mean()
    agg_dict['min_sharpness_90pct'] = df['sharpness_90pct'].min()
    agg_dict['max_sharpness_90pct'] = df['sharpness_90pct'].max()

    if kmeans_loaded is not None:
        agg_dict['yiq_clusters'] = kmeans_loaded.predict(df[['img_y', 'img_i', 'img_q']])
        agg_dict['yiq_clusters_top2'] = agg_dict['yiq_clusters'][:2]
    else:
        agg_dict['yiq_clusters'] = None
        agg_dict['yiq_clusters_top2'] = None

    return agg_dict


def process_listing_dir(base_path, listing_dir, out_base_path):
    # Ensure that it contains photos and output is not already done
    listing_path = os.path.join(base_path, listing_dir)
    out_file = os.path.join(out_base_path, '%s.csv' % listing_dir)

    try:
        listing_id = int(listing_dir)
        img_files = [os.path.join(listing_path, fname) for fname in os.listdir(listing_path)]
        if len(img_files) == 0:
            return None
    except (OSError, ValueError) as e:
        logging.error('Error: %s' % e)
        return None

    df = get_image_feat_df(listing_id, img_files, out_file)
    if df is None:
        return None
    agg_dict = aggregate_image_feats(df)
    agg_dict['listing_id'] = listing_id
    return agg_dict


def worker_run(worker_id, image_base_path, listing_dirs, feat_out_base_path, agg_out_base_path):
    worker_out_file = os.path.join(agg_out_base_path, '%s.json' % worker_id)
    aggs = []
    for listing_dir in listing_dirs:
        agg_dict = process_listing_dir(image_base_path, listing_dir, feat_out_base_path)
        if agg_dict is not None:
            aggs.append(agg_dict)
    df = pd.DataFrame(aggs)
    logging.info(df.head(1).transpose())
    df.to_json(worker_out_file, orient='split')


def run(n_workers, image_base_path, feat_out_base_path, agg_out_base_path):
    if os.path.exists(agg_out_base_path):
        logging.info('Wiping out %s.' % agg_out_base_path)
        shutil.rmtree(agg_out_base_path)
    os.makedirs(agg_out_base_path)

    if not os.path.exists(feat_out_base_path):
        os.makedirs(feat_out_base_path)

    jobs = []
    listing_dirs = list(os.listdir(image_base_path))
    dirs_per_worker = len(listing_dirs) / n_workers

    for i in xrange(0, len(listing_dirs), dirs_per_worker):
        dir_slice = listing_dirs[i:i + dirs_per_worker]
        logging.info('%d, %s' % (i, dir_slice))

        p = multiprocessing.Process(target=worker_run, args=(
            str(i), image_base_path, dir_slice, feat_out_base_path, agg_out_base_path))
        jobs.append(p)
        p.start()
    return jobs


def main():
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='generate_image_feats_%s.log' % time.strftime("%Y%m%d-%H%M%S"))

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", default=7, type=int, help="number of workers to spawn")
    parser.add_argument("--image-bgit ase-path", default='../data/images_sample/', help="path to photos")
    parser.add_argument("--feat-out-base-path", default='../build/photo-features/tmp/', help="intermediate output dir")
    parser.add_argument("--agg-out-base-path", default='../build/photo-features/aggs/', help="worker output dir")
    args = parser.parse_args()

    image_base_path = args.image_base_path
    feat_out_base_path = args.feat_out_base_path
    agg_out_base_path = args.agg_out_base_path

    start = time.time()
    jobs = run(args.n_workers, image_base_path, feat_out_base_path, agg_out_base_path)

    for job in jobs:
        job.join()
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))


if __name__ == "__main__":
    main()
