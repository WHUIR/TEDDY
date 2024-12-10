import torch
import torch.nn as nn
import torch.nn.functional as F
from module import  Disentangel_mask_hard, Disentangel_mask_soft, Seq_mask_last_k, Seq_mask_kth, Trend_interest_TemporalConvNet, Trend_interest_transforemer_block, Prediction_inner, MLP_diversity_rep, Soft_attention_diversity, Linear_Max_pooling_all_diversity, Linear_diversity


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Disentangle_interest(nn.Module):
    def __init__(self, args):
        super(Disentangle_interest, self).__init__()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.item_num = args.item_count+1
        self.max_len = args.max_len
        self.position_embedding_flag = args.position_embedding_flag
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)
        # self.threshold_embedding = nn.Embedding(self.item_num, 1)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.norm_trend_rep = LayerNorm(self.hidden_size)
        self.norm_diversity_rep = LayerNorm(self.hidden_size) 

        self.seq_rep_mask_k = Seq_mask_last_k(self.hidden_size, k=1)  ## k_means
        # self.seq_rep_mask_k = Seq_mask_kth(self.hidden_size, k=3)  ## K_th
        self.disentangle_mask_hard = Disentangel_mask_hard(self.device, self.hidden_size)
        # self.disentangle_mask_soft = Disentangel_mask_soft()
        self.mask_threshold = 0.2
        
        
        self.tcn_trend_interest = Trend_interest_TemporalConvNet(args)  ## TEDDY
        # self.transformer_trend_interest = Trend_interest_transforemer_block(args)  ## Ablation
        self.trend_seq_rep_mask_k = Seq_mask_last_k(self.hidden_size, k=1)

        self.predict_layer_inner_trend = Prediction_inner()
        self.mlp_diversity_rep = MLP_diversity_rep(self.hidden_size)
        # self.diversity_rep_soft_attention = Soft_attention_diversity(self.hidden_size, self.device)
        
        # self.diversity_seq_rep_mask_k = Seq_mask_last_k(self.hidden_size, k=1)
        self.linear_max_pooling_all_diversity = Linear_Max_pooling_all_diversity(self.max_len, self.hidden_size, self.item_num)
        # self.linear_diversity = Linear_diversity(self.hidden_size, self.item_num)

        self.score_trend_lambda = args.score_trend_lambda
        self.ce_loss = nn.CrossEntropyLoss()
        # self.softmax_score = nn.Softmax(dim=-1)
        # self.max_pooling_scores = nn.MaxPool1d(2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.position_embedding.weight)
        # nn.init.xavier_normal_(self.threshold_embedding.weight)
        
        
    def forward(self, inputs):
        if self.position_embedding_flag:
            emb = self.item_embedding(inputs)
            pos_embedding = self.position_embedding.weight.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
            emb = self.embed_dropout(emb + pos_embedding)
        else:
            emb = self.embed_dropout(self.item_embedding(inputs))
            
        mask_input = (inputs > 0).float()
        rep_seq = self.seq_rep_mask_k(emb)
        mask_trend, mask_diversity = self.disentangle_mask_hard(rep_seq, emb, self.mask_threshold, mask_input)
     
        trend_emb = emb * mask_trend.unsqueeze(-1)
        diversity_emb = emb * mask_diversity.unsqueeze(-1)

        trend_rep, hn_rep = self.tcn_trend_interest(trend_emb.transpose(1, 2))
        # trend_rep = self.transformer_trend_interest(trend_emb, mask_input.unsqueeze(1).repeat(1, inputs.shape[1], 1).unsqueeze(1)).transpose(1, 2)  ## ablation transformer
        
        trend_rep = self.dropout(self.norm_trend_rep(trend_rep.transpose(1, 2)))  ## should be use
        trend_seq_rep = self.trend_seq_rep_mask_k(trend_rep)
        
        scores_trend = self.predict_layer_inner_trend(trend_seq_rep, self.item_embedding.weight)
        
        diversity_rep = self.mlp_diversity_rep(emb)
        # diversity_rep = self.diversity_rep_soft_attention(diversity_emb, mask_input)
        diversity_rep = self.dropout(self.norm_diversity_rep(diversity_rep))
        
        scores_diversity = self.linear_max_pooling_all_diversity(diversity_rep, mask_diversity.unsqueeze(-1))
        # scores_diversity = self.linear_diversity(diversity_rep)

        scores_final = self.score_trend_lambda * scores_trend + (1-self.score_trend_lambda) * scores_diversity
        return scores_final, scores_trend, scores_diversity  


