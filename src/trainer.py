import torch.nn as nn
import torch
import torch.optim as optim
import datetime
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import pickle


def plot_adj(adj_mat, list_items):
    sns.heatmap(adj_mat, cmap='Reds', linewidths=.5,
                xticklabels=list_items,
                yticklabels=list_items, vmax=0.015)
    sns.set(font_scale=1.4)
    # ax.xaxis.tick_top()
    plt.show()
    quit()


def cor_seq(seq_temp, model):
    emb = model.item_embedding(seq_temp)
    mask_input = (seq_temp > 0).float()

    rep_seq = model.seq_rep_mask_k(emb)
    mask_trend, mask_diversity = model.disentangle_mask_hard(rep_seq, emb, model.mask_threshold, mask_input)

    trends = emb[mask_trend == 1][:4]
    diversitys = emb[mask_diversity == 1]

    # trend_rep = model.tcn_trend_interest(emb.transpose(1, 2)).transpose(1, 2)
    # trend_rep = model.norm_trend_rep(trend_rep).squeeze(0)
    # diversity_rep = model.mlp_diversity_rep(emb)
    # diversity_rep = model.norm_diversity_rep(diversity_rep).squeeze(0)

    # trend_rep = trend_rep[mask_trend.squeeze() == 1]
    # diversity_rep = diversity_rep[mask_diversity.squeeze() == 1]

    adj_trend = torch.matmul(trends, trends.t())
    adj_diversity = torch.matmul(diversitys, diversitys.t())

    trend_topk = torch.abs(adj_trend[:4, :4])

    zeros_mat = torch.zeros([adj_diversity.shape[0] + trend_topk.shape[0], adj_diversity.shape[0] + trend_topk.shape[0]])
    zeros_mat[:adj_diversity.shape[0], :adj_diversity.shape[0]] = adj_diversity
    zeros_mat[adj_diversity.shape[0]:, adj_diversity.shape[0]:] = trend_topk
    list_item = torch.cat([diversitys, trends], dim=0).tolist()
    plot_adj(zeros_mat, list_item)


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_recall(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return recall


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def recalls_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_recall(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['Recall@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def calculate_metrics(model, val_batch, metric_ks):
    seqs, labels = val_batch
    scores, scores_trend, scores_diversity, = model(seqs)  # B x V
    # socres = scores_diversity
    metrics = recalls_and_ndcgs_k(scores, labels, metric_ks)
    return metrics


def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model = nn.DataParallel(model)
    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_Recall@5': 0, 'Best_NDCG@5': 0, 'Best_Recall@10': 0, 'Best_NDCG@10': 0, 'Best_Recall@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_Recall@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_Recall@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_Recall@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model.train()
        # lr_scheduler.step()
        flag_update = 0
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            logits, logits_trend, logits_diversity = model(train_batch[0])
            labels = train_batch[1].squeeze(-1)
            loss = model.ce_loss(logits, labels)
            loss.backward()
            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
            # quit()
            optimizer.step()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss.item()))
        print('start predicting: ', datetime.datetime.now())
        logger.info('start predicting: {}'.format(datetime.datetime.now()))
        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': []}
            # metrics_dict_mean = {}
            for val_batch in val_data_loader:
                val_batch = [x.to(device) for x in val_batch]
                metrics = calculate_metrics(model, val_batch, metric_ks)
                for k, v in metrics.items():
                    metrics_dict[k].append(v)
        
        for key_temp, values_temp in metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            if values_mean > best_metrics_dict['Best_' + key_temp]:
                flag_update = 1
                bad_count = 0
                best_metrics_dict['Best_' + key_temp] = values_mean
                best_epoch['Best_epoch_' + key_temp] = epoch_temp
                best_model = copy.deepcopy(model)
               
        if flag_update == 0:
            bad_count += 1
        else:
            print(best_metrics_dict)
            print(best_epoch)
            logger.info(best_metrics_dict)
            logger.info(best_epoch)
      
        if bad_count >= args.patience:
            break
        with torch.no_grad():
            test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': []}
            test_metrics_dict_mean = {}
            for test_batch in test_data_loader:
                test_batch = [x.to(device) for x in test_batch]
                
                metrics = calculate_metrics(best_model, test_batch, metric_ks)
                for k, v in metrics.items():
                    test_metrics_dict[k].append(v)
        for key_temp, values_temp in test_metrics_dict.items():
            values_mean = round(np.mean(values_temp) * 100, 4)
            test_metrics_dict_mean[key_temp] = values_mean
        print('---------------------------Test------------------------------------------------------')
        logger.info('--------------------------Test------------------------------')
        print(test_metrics_dict_mean)
        logger.info(test_metrics_dict_mean)


    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    with torch.no_grad():
        test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            metrics = calculate_metrics(best_model, test_batch, metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    return test_metrics_dict_mean, best_model


def cort_plot(batch):
    seqs, candidates, labels = batch
    seqs_len = torch.sum((seqs > 0).float(), dim=-1)
    index_20 = [inx for inx, i in enumerate(seqs_len) if i >= 20]
    cor_seq(seqs[index_20[0]].unsqueeze(0), model)


def mask_popular_dist(model, test_batch, item_dict_dist):
    seqs, labels = test_batch
    emb = model.item_embedding(seqs)
    mask_input = (seqs > 0).float()
    rep_seq = model.seq_rep_mask_k(emb)
    mask_trend, mask_diversity = model.disentangle_mask_hard(rep_seq, emb, model.mask_threshold, mask_input)
    for seq_temp, mask_trend_temp, mask_diversity_temp, mask_temp in zip(seqs, mask_trend, mask_diversity, mask_input):
        num_items = torch.sum(mask_temp)
        for i in range(int(num_items.item())):
            item_temp = int(seq_temp[-(i+1)].item())
            mask_item = mask_trend_temp[-(i+1)]
            if item_temp not in item_dict_dist:
                item_dict_dist[item_temp] = [int(mask_item.item())]
            else:
                item_dict_dist[item_temp].append(int(mask_item.item()))
    return item_dict_dist


def len_trend_pop(model, test_batch, len_trend_prop_dist):
    seqs, labels = test_batch
    emb = model.item_embedding(seqs)
    mask_input = (seqs > 0).float()
    rep_seq = model.seq_rep_mask_k(emb)
    mask_trend, mask_diversity = model.disentangle_mask_hard(rep_seq, emb, model.mask_threshold, mask_input)
    for seq_temp, mask_trend_temp, mask_diversity_temp, mask_temp in zip(seqs, mask_trend, mask_diversity, mask_input):
        num_items = int(torch.sum(mask_temp).item())
        num_trend_items = torch.sum(mask_trend_temp).item()
        if num_items not in len_trend_prop_dist: 
            len_trend_prop_dist[num_items] = [num_trend_items/num_items]
        else:
            len_trend_prop_dist[num_items].append(num_trend_items/num_items)
    return len_trend_prop_dist


def inference(model, test_data_loader, metric_ks, device, epoch_temp):
    model = model.to(device)
    model.eval()
    masks_list, mask_trend_list, mask_diversity_list = [], [], []
    item_dict_dist = {}
    with torch.no_grad():
        test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        len_trend_prop_dist = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            # item_dict_dist = mask_popular_dist(model, test_batch, item_dict_dist)
            len_trend_prop_dist = len_trend_pop(model, test_batch, len_trend_prop_dist)
    cort_plot(test_batch)
    quit()
    with open('amazon_beauty_len_trend_pop_dist.pkl', 'wb') as f:
        pickle.dump(len_trend_prop_dist, f)
    quit()

    with open('amazon_beauty_item_dist.pkl', 'wb') as f:
        pickle.dump(item_dict_dist, f)

            # cort_plot(test_batch)
    #         metrics = calculate_metrics(model, test_batch, metric_ks)
    #         masks, masks_trend, masks_diversity = mask_distribution(model, test_batch)
    #         masks_list += masks.to('cpu').tolist()
    #         mask_trend_list += masks_trend.to('cpu').tolist()
    #         mask_diversity_list += masks_diversity.to('cpu').tolist()
    #         for k, v in metrics.items():
    #             test_metrics_dict[k].append(v)
    # # plot_dist_mask(masks_list, mask_trend_list, mask_diversity_list)
    #
    # for key_temp, values_temp in test_metrics_dict.items():
    #     values_mean = round(np.mean(values_temp) * 100, 4)
    #     test_metrics_dict_mean[key_temp] = values_mean
    #
    # print(test_metrics_dict_mean)
