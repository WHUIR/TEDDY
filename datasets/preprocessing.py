import os
from urllib import request
import sys
import zipfile
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from tqdm import trange
tqdm.pandas()



dataset = 'steam'  ## [movielens-1m, movielens-20m, amazon_beauty, amazon_musical, amazon_toys, steam, yelp, lastfm]


def _progress(block_num, block_size, total_size):
    '''回调函数
       @block_num: 已经下载的数据块
       @block_size: 数据块的大小
       @total_size: 远程文件的大小
    '''
    sys.stdout.write('\r>> Downloading %.1f%%' % (float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()


def download_data(dataset):
    path_save_dir = os.getcwd() + '\\' + dataset
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    if dataset == 'movielens-1m':
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        if not os.path.exists(os.path.join(path_save_dir, 'ml-1m')):
            print('Download ml-1m.zip')
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ml-1m.zip'), _progress)
            print()
            print('Download Done and Unzip')
            zip = zipfile.ZipFile(os.path.join(path_save_dir, 'ml-1m.zip'))
            zip.extractall(path_save_dir)
            zip.close()
            os.remove(os.path.join(path_save_dir, 'ml-1m.zip'))
            print('Done')
        else:
            print('Movielens-1m raw data already exist.')
    elif dataset == 'movielens-20m':
        url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
        if not os.path.exists(os.path.join(path_save_dir, 'ml-20m')):
            print('Download ml-20m.zip')
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ml-20m.zip'), _progress)
            print()
            print('Download Done and Unzip')
            zip = zipfile.ZipFile(os.path.join(path_save_dir, 'ml-20m.zip'))
            zip.extractall(path_save_dir)
            zip.close()
            os.remove(os.path.join(path_save_dir, 'ml-20m.zip'))
            print('Done')
        else:
            print('Dataset already exist.')
    elif dataset == 'amazon_beauty':
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv'
        if not os.path.exists(os.path.join(path_save_dir, 'ratings_Beauty.csv')):
            print('Download Amazon-Beauty.csv')
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ratings_Beauty.csv'), _progress)
            print()
            print('Download Done.')
        else:
            print('Dataset already exist.')
    elif dataset == 'amazon_musical':
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Musical_Instruments.csv'
        if not os.path.exists(os.path.join(path_save_dir, 'ratings_Musical.csv')):
            print('Download Amazon-Musical.csv')
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ratings_Musical.csv'), _progress)
            print()
            print('Download Done.')
    elif dataset == 'amazon_toys':
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv'
        if not os.path.exists(os.path.join(path_save_dir, 'ratings_Musical.csv')):
            print('Download Amazon-Toys-and-Games.csv')
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ratings_Toys.csv'), _progress)
            print()
            print('Download Done.')
        else:
            print('Dataset already exist.')
    elif dataset == 'amazon_baby':
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Baby.csv'
        if not os.path.exists(os.path.join(path_save_dir, 'ratings_Baby.csv')):
            print('Download Amazon-Baby.csv')
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ratings_Baby.csv'), _progress)
            print()
            print('Download Done.')
        else:
            print('Dataset already exist.')
    elif dataset == 'amazon_books':
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv'
        if not os.path.exists(os.path.join(path_save_dir, 'ratings_Baby.csv')):
            print('Download Amazon-Books.csv')
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ratings_Books.csv'), _progress)
            print()
            print('Download Done.')
        else:
            print('Dataset already exist.')
    elif dataset == 'steam':
        url = 'http://deepx.ucsd.edu/public/jmcauley/steam/australian_users_items.json.gz'
        if not os.path.exists(os.path.join(path_save_dir, 'ratings_Steam.csv')):
            print('Download Steam')
            quit()
            response = request.urlretrieve(url, os.path.join(path_save_dir, 'ratings_steam.csv'), _progress)
            print()
            print('Download Done.')
        else:
            print('Dataset already exist.')
    elif dataset == 'yelp':
        if not os.path.exists(os.path.join(path_save_dir, 'ratings_Yelp.csv')):
            path_yelp = ''
            user_seq_dict = {}
            with open(os.path.join(path_yelp, 'train.txt'), 'r') as f:
                lines = f.readlines()
                for line_temp in lines:
                    line_temp = line_temp.strip('\n').split(' ')
                    user_seq_dict[int(line_temp[0])] = [int(i) for i in line_temp[1:]]
            with open(os.path.join(path_yelp, 'test.txt'), 'r') as f:
                lines = f.readlines()
                for line_temp in lines:
                    line_temp = line_temp.strip('\n').split(' ')
                    if int(line_temp[0]) in user_seq_dict:
                        user_seq_dict[int(line_temp[0])] += [int(i) for i in line_temp[1:]]
                    else:
                        user_seq_dict[int(line_temp[0])] = [int(i) for i in line_temp[1:]]
            users, items = [], []
            for user_temp in user_seq_dict:
                for item_temp in user_seq_dict[user_temp]:
                    users.append(user_temp)
                    items.append(item_temp)
            pd_temp = pd.DataFrame({'uid': users, 'sid': items})
            pd_temp.to_csv(os.path.join(path_save_dir, 'ratings_Yelp.csv'), index=False, header=False)

        else:
            print('Dataset already exist.')


def make_implicit(df):
    print('Turning into implicit ratings.')
    min_rating = 4
    print('Min rating: {}'.format(min_rating))
    df = df[df['rating'] >= min_rating]
    return df


def filter_triplets(df):
    print('Filtering tripltes.')
    min_sc = 5
    min_uc = 5
    print('Min item : {}'.format(min_sc))
    print('Min user interaction: {}'.format(min_uc))
    if min_sc > 0:
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_sc]
        df = df[df['sid'].isin(good_items)]
    if min_uc > 0:
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_uc]
        df = df[df['uid'].isin(good_users)]
    return df


def densify_index(df):
    print('Desifying index.')
    umap = {u: (i+1) for i, u in enumerate(set(df['uid']))}
    smap = {s: (i+1) for i, s in enumerate(set(df['sid']))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)
    return df, umap, smap


def split_df(df, user_count):
    print('Splitting: Leave One Out.')
    user_group = df.groupby('uid')
    user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
    train, val, test = {}, {}, {}
    for user in range(1, user_count+1):
        items = user2items[user]
        train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
    return train, val, test


def split_withno_time(df, user_count):
    print('Splitting: Leave One Out.')
    user_group = df.groupby('uid')
    train, val, test = {}, {}, {}
    for user, group_temp in zip(range(1, user_count+1), user_group):
        user_temp = group_temp[0]
        seq_temp = group_temp[-1]['sid'].tolist()
        train[user_temp], val[user_temp], test[user_temp] = seq_temp[:-2], seq_temp[-2:-1], seq_temp[-1:]
    return train, val, test


def data_preprocess(dataset):
    if dataset == 'movielens-1m':
        path_save_dir = os.getcwd() + '\\' + dataset + '\\' + 'ml-1m'
        # df = pd.read_csv(os.path.join(path_save_dir, 'ratings.dat'), sep='::', header=None, nrows=10000)
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings.dat'), sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = make_implicit(df)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        train, val, test = split_df(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'movielens-20m':
        path_save_dir = os.getcwd() + '\\' + dataset + '\\' + 'ml-20m'
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings.csv'), header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        df['rating'] = pd.to_numeric(df['rating'])
        df = make_implicit(df)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        train, val, test = split_df(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'amazon_beauty':
        path_save_dir = os.getcwd() + '\\' + dataset
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings_Beauty.csv'), header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        df = make_implicit(df)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        print(len(umap))

        train, val, test = split_df(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'amazon_musical':
        path_save_dir = os.getcwd() + '\\' + dataset
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings_Musical.csv'), header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        df = make_implicit(df)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        print('Item numbers: {}'.format(len(set(list(df['sid'])))))
        train, val, test = split_df(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'amazon_toys':
        path_save_dir = os.getcwd() + '\\' + dataset
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings_Toys.csv'), header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        df = make_implicit(df)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        print('Item numbers: {}'.format(len(set(list(df['sid'])))))
        train, val, test = split_df(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'amazon_baby':
        path_save_dir = os.getcwd() + '\\' + dataset
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings_Baby.csv'), header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        df = make_implicit(df)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        print('Item numbers: {}'.format(len(set(list(df['sid'])))))
        train, val, test = split_df(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'amazon_books':
        path_save_dir = os.getcwd() + '\\' + dataset
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings_Books.csv'), header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        df = make_implicit(df)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        print('Item numbers: {}'.format(len(set(list(df['sid'])))))
        train, val, test = split_df(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'steam':
        path_save_dir = os.getcwd() + '/' + dataset
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings_Steam.csv'), header=None)
        
        df.columns = ['uid', 'sid']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        print(df)
        print(len(set(list(df['sid']))))
        quit()
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        print('Item numbers: {}'.format(len(set(list(df['sid'])))))
        train, val, test = split_withno_time(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'yelp':
        path_save_dir = os.getcwd() + '\\' + dataset
        df = pd.read_csv(os.path.join(path_save_dir, 'ratings_Yelp.csv'), header=None)
        df.columns = ['uid', 'sid']
        df = df.drop(0)
        df = df.reset_index(drop=True)
        df = filter_triplets(df)
        df, umap, smap = densify_index(df)
        print('Item numbers: {}'.format(len(set(list(df['sid'])))))
        train, val, test = split_withno_time(df, len(umap))
        data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    elif dataset == 'lastfm':
        path = 'lastfm/all_data.csv'
        df = pd.read_csv(path)
        df.columns = ['uid',]
        quit()
    else:
        data_all = None
    return data_all


def save_data(data_all):
    path_save_dir = os.getcwd() + '\\' + dataset + '\\preprocess'
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    if not os.path.exists(os.path.join(path_save_dir, 'dataset_5_5.pkl')):
        with open(os.path.join(path_save_dir, 'dataset_5_5.pkl'), 'wb') as f:
            pickle.dump(data_all, f)
        print()
        print('Preprocess Data Saved.')
    else:
        print()
        print('Preprocess data already exist.')


def negative_samples_generate(data_all, radom_seed=1997):
    np.random.seed(radom_seed)
    negative_samples = {}
    user_count = len(data_all['umap'])
    sample_size = 100
    item_count = len(data_all['smap'])
    print('Sampling negative items with size {}.'.format(sample_size))
    for user in trange(1, user_count+1):
        if isinstance(data_all['train'][user][1], tuple):
            seen = set(x[0] for x in data_all['train'][user])
            seen.update(x[0] for x in data_all['val'][user])
            seen.update(x[0] for x in data_all['test'][user])
        else:
            seen = set(data_all['train'][user])
            seen.update(data_all['val'][user])
            seen.update(data_all['test'][user])
        samples = []
        for _ in range(sample_size):
            item = np.random.choice(item_count) + 1
            while item in seen or item in samples:
                item = np.random.choice(item_count) + 1
            samples.append(item)
        negative_samples[user] = samples
    print('Sampling Done.')
    return negative_samples


def save_negative_samples(negative_samples):
    path_save_dir = os.getcwd() + '\\' + dataset + '\\preprocess'
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    if not os.path.exists(os.path.join(path_save_dir, 'random_negative_samples_test_100_5_5.pkl')):
        with open(os.path.join(path_save_dir, 'random_negative_samples_test_100_5_5.pkl'), 'wb') as f:
            pickle.dump(negative_samples, f)
        print('Negative Samples Saved.')
    else:
        print()
        print('Negative samples already exist.')


def main():
    # download_data(dataset)

    pd_dataset = data_preprocess(dataset)
    quit()
    save_data(pd_dataset)
    negative_samples_test = negative_samples_generate(pd_dataset, radom_seed=1997)
    save_negative_samples(negative_samples_test)


if __name__ == '__main__':
    main()
