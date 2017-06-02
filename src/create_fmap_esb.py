"""
Adapted from https://github.com/jeongyoonlee/kaggler-template
"""
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-models', required=True, nargs='+',
                        dest='base_models')
    parser.add_argument('--feature-map-file', required=True,
                        dest='feature_map_file')

    args = parser.parse_args()

    i = 0
    with open(args.feature_map_file, 'w') as f:
        for model in args.base_models:
            for col in ['high', 'medium', 'low']:
                f.write('{}\t{}\tq\n'.format(i, model + '_' + col))
                i += 1
