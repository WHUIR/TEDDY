import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_Val_Test
from model import Disentangle_interest
from trainer import model_train, inference
import os
import logging
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='beauty', help='Dataset name: movielens-1m, movielens-20m, beauty, amazon_beauty, steam, yelp, amazon_musical')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=2022, help='Random seed')  
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--item_count', type=int, default=3533, help='The total number of items in raw dataset')  # ['movielens-1m': 3533, 'movielens-20m': 27198, 'Beauty': 95313]
parser.add_argument('--position_embedding_flag', type=str, default=False, help='Position embedding switch')
# parser.add_argument('--user_count', type=int, default=None, help='The total number of users in raw dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')  ## 1024
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden Size')
parser.add_argument('--num_blocks', type=int, default=1, help='Number of Trend interests blocks')
parser.add_argument('--kernel_size', type=int, default=4, help='Kernel size of CausalCNN')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--score_trend_lambda', default=0.2, help='Weight of prediction score from trend term')  ## 0.8
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of embedding')
parser.add_argument('--tied_weights', type=str, default=True, help='tie the word embedding and softmax weights (default: True)')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')  ## 100
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop')
parser.add_argument('--description', type=str, default='beauty last 3th item for adaptive mask', help='Model briefly introduction')
parser.add_argument('--training_flag', default=True, help='Inference or Training, True for Training, otherwise')
args = parser.parse_args()
print(args)


if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)

if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset)
if not os.path.exists('model_dict/' + args.dataset):
    os.makedirs('model_dict/' + args.dataset)


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    # random_seed = random.randint(0, 100000)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


fix_random_seed_as(args.random_seed)


def item_count():
    dict_item_count = {'movielens-1m': 3125, 'movielens-20m': 27198, 'amazon_beauty': 41795, 'amazon_musical': 9608, 'steam': 12030, 'yelp': 38048, 'amazon_baby': 14361}   # Amazon_beauty: 41795 (5,5); 12101 
    if args.dataset in dict_item_count:
        args.item_count = dict_item_count[args.dataset]
    else:
        raise Exception('Dataset-{} is not included in the options'.format(args.dataset))


def user_count():
    dict_user_count = {'movielens-1m': 71, 'amazon_beauty': 35932}  ## movielens_1m_sample: 71
    if args.dataset in dict_user_count:
        args.user_count = dict_user_count[args.dataset]
    else:
        raise Exception('Dataset-{} is not included in the options'.format(args.dataset))


def model_save(model, dataset, results):
    path_model = 'model_dict/' + dataset + '/' + str(results) + '_' + str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())) + '.pt'
    torch.save(model.state_dict(), path_model)


def bibar_plot(data_raw):
    pop_trend_dict = pickle.load(open('amazon_beauty_len_trend_pop_dist.pkl', 'rb'))
    mean_trend_dict = {}
    lens = [len(seq_temp)+1 for seq_temp in data_raw.values()]
    len_dict = Counter(lens)
    freq_len_dict = {}
    seqs_num = len(data_raw)
    count_10 = 0
    for num_len in len_dict:
        if num_len < 10:
            count_10 += len_dict[num_len]
    print(count_10/seqs_num)
    quit()
    for item_temp in pop_trend_dict:
        mean_trend_dict[item_temp] = np.mean(pop_trend_dict[item_temp])
    for i in range(4, 50):
        freq_len_dict[i] = (mean_trend_dict[i], len_dict[i]/seqs_num)
    all = []
    a = []
    b = []
    for i in range(1, 51):  
        if i%5 == 0:
            if i in freq_len_dict:
                a.append(freq_len_dict[i][0])
                b.append(freq_len_dict[i][1])
                all.append([np.round(np.mean(a), 4), np.round(np.sum(b), 4)])
                a = []
                b = []
        else: 
            if i in freq_len_dict:
                a.append(freq_len_dict[i][0])
                b.append(freq_len_dict[i][1])
    
    all.append([np.round(np.mean(a), 4), np.round(np.sum(b), 4)])
    
    
def dist_len_pop_plot():
    pop_trend_dict = pickle.load(open('amazon_beauty_len_trend_pop_dist.pkl', 'rb'))
    points = []
    for len_temp in pop_trend_dict:
        for pop_temp in pop_trend_dict[len_temp]:
            points.append((len_temp, pop_temp))
    dict_z = {}
    for point_temp in points:
        if point_temp in dict_z:
            dict_z[point_temp] += 1
        else:
            dict_z[point_temp] = 1
    
    x = np.array([temp[0] for temp in dict_z.keys()])
    y = np.array([temp[1] for temp in dict_z.keys()])
    z = np.array([min(temp, 50) for temp in dict_z.values()])
    
    # sns.scatterplot(x=x, y=y, s=1, color=".15")
    # sns.histplot(x=x, y=y, bins=500, pthresh=.1, cmap="mako")
    # sns.kdeplot(x=x, y=y, levels=10, color="r", linewidths=1)

    ct = plt.tricontour(x,y,z, 85, cmap='Greens_r', alpha=0.3)     
    ctf = plt.tricontourf(x,y,z, 85, cmap='Greens_r', alpha=0.3)  
    # sns.kdeplot(x=x, y=y, fill=True, cmap='Spectral', cbar=True)
    cbar = plt.colorbar(ctf)    

    # xy = np.vstack([x,y])  
    # z = gaussian_kde(xy)(xy)

    # # Sort the points by density, so that the densest points are plotted last
    # idx = z.argsort()
    # x, y, z = x[idx], y[idx], z[idx]

    # fig, ax = plt.subplots()
    plt.scatter(x, y, c=z, s=1, cmap='Spectral', alpha=0.4) 
    # sns.regplot(x=x, y=y, fit_reg = False, x_jitter = 0.2, scatter_kws = {'alpha' : 1/3})
    # plt.colorbar()
    plt.ylim(0, 1.01)
    plt.xlim(4, 51)
    plt.xticks([int(i) for i in np.linspace(4, 52, 13)], fontweight='bold', fontsize=10)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], fontweight='bold', fontsize=10)
    plt.xlabel('Length of Sequence', fontweight='bold', fontsize=10)
    plt.ylabel('Proportion of Items w.r.t Interest Trend', fontweight='bold', fontsize=10)
    # # plt.scatter(x, y, s=1, alpha=0.5)
    plt.grid(color='black', linestyle='--', linewidth=1,alpha=0.2)
    plt.savefig('counter_scater_bar.png', transparent=True)
    plt.show()


def dist_item_plot():
    item_trend_dict = pickle.load(open('amazon_beauty_item_dist.pkl', 'rb'))
    path_data = '../BERT4Rec/' + args.dataset + '/preprocess/dataset_5_5.pkl'
    data_raw = pickle.load(open(path_data, 'rb'))

    item_count_dict = {}
    for seq_temp in data_raw['train'].values():
        for item_temp in seq_temp:
            if item_temp in item_count_dict:
                item_count_dict[item_temp] += 1
            else:
                item_count_dict[item_temp] = 1
    item_combine_dict = {}
    for item_temp in item_count_dict:
        if item_temp in item_trend_dict:
            rat_temp = np.sum(item_trend_dict[item_temp])/len(item_trend_dict[item_temp])
            item_combine_dict[item_temp] = (item_count_dict[item_temp], rat_temp)  
    
    x = np.array([temp[0] for temp in item_combine_dict.values()])
    y = np.array([temp[1] for temp in item_combine_dict.values()])
    
    # plt.scatter(x, y, s=1, alpha=0.5)
    # plt.xlim(0, 100)
    # plt.savefig('dist_p.png')
    # plt.show()
    # quit()

    gride_y = np.linspace(0, 1.1, 11)
    gride_x = np.linspace(0, 101, 11)
    # gride_x = np.append(gride_x, 500)
    gride_num = {}
    array_list = []
    for i in range(len(gride_x)-1):
        temp_y_list = []
        for j in range(len(gride_y)-1):
            count=0 
            for point_temp in item_combine_dict.values():
                if gride_x[i] <= point_temp[0] < gride_x[i+1]:
                    temp_grid_x = gride_x[i]
                    if gride_y[j] <= point_temp[1] < gride_y[j+1]:
                        temp_grid_y = gride_y[j]
                        count+=1
                        if (temp_grid_x, temp_grid_y) not in gride_num:
                            gride_num[(temp_grid_x, temp_grid_y)] = 1 
                        else:
                            gride_num[(temp_grid_x, temp_grid_y)] += 1 
            temp_y_list.append(count)
        array_list.append(temp_y_list)
    array_back = np.array(array_list).transpose()
    array_back = np.flipud(array_back)
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(array_back, cmap=cmap, vmax=100, center=0, square=True, linewidths=.0000000000005, cbar_kws={"shrink": .5})

    plt.scatter(x, y, s=1, alpha=0.5)
    sns.kdeplot(x=x, y=y, fill=True, cmap='Spectral', cbar=True)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    # sns.kdeplot(x=x, y=y, shade=True, bw="silverman", gridsize=5000, clip=(0, 500),  cmap="Purples")
    # sns.kdeplot(x, y, shade=True, bw="silverman", gridsize=50000, clip=(0, 500),  cmap="Purples")

    # Calculate the point density
    # f, ax = plt.subplots(figsize=(6, 6))
    # sns.scatterplot(x=x, y=y, s=5, color=".15")
    # sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    # sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
    plt.savefig('dist_3.png')
    plt.show()


def main():
    # item_count()
    # user_count()
    path_data = 'datasets/' + args.dataset + '/dataset.pkl'
    
    data_raw = pickle.load(open(path_data, 'rb'))
    args.item_count = len(data_raw['smap'])
    
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    model = Disentangle_interest(args)
    if args.training_flag:
        model_best, test_results = model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger)
        # model_save(model_best, args.dataset, test_results['Recall@10'])
    else:
        # bibar_plot(data_raw['train'])
        # dist_len_pop_plot()
        # dist_item_plot()
        path_model = 'model_dict/' + args.dataset + '/'
        model.load_state_dict(torch.load('model_dict/amazon_beauty/' + '.pt'))
        inference(model, test_data_loader, args.metric_ks, args.device, epoch_temp=0)
    print(args)


if __name__ == '__main__':
    main()
