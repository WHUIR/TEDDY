import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class Disentangel_mask_soft(nn.Module):
    def __init__(self, ):
        super(Disentangel_mask_soft, self).__init__()
        self.gate_linear = nn.Linear(1, 2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.gate_linear.weight)

    def forward(self, seq_rep, seqs, mask_threshold, mask_inputs):
        alpha_score = torch.matmul(seqs, seq_rep.unsqueeze(-1))
        masks = F.gumbel_softmax(self.gate_linear((alpha_score - mask_threshold)), hard=True)[:, :, 0]
        return masks * mask_inputs, (1-masks) * mask_inputs


class Disentangel_mask_hard(nn.Module):
    def __init__(self, device, hidden_size):
        super(Disentangel_mask_hard, self).__init__()
        self.device = device
        # self.seq_linear = nn.Linear(hidden_size, hidden_size)
        # self.seqs_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, seq_rep, seqs, mask_threshold, mask_inputs):
        # seq_rep = torch.relu(self.seq_linear(seq_rep))
        # seqs = torch.relu(self.seqs_linear(seqs))
        seqs_norm = seqs/seqs.norm(dim=-1)[:, :, None]
        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        alpha_score = torch.matmul(seqs_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1) - mask_threshold
        
        # alpha_score = torch.abs(torch.matmul(seqs_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1) - mask_threshold)
        mask_one = torch.where(alpha_score > 0, torch.ones_like(alpha_score).to(self.device), alpha_score)
        mask_onehot = torch.where(mask_one < 0, torch.zeros_like(alpha_score).to(self.device), mask_one)
        mask_onehot = mask_onehot - alpha_score.detach() + alpha_score
        return mask_onehot * mask_inputs, (1-mask_onehot) * mask_inputs


class Seq_mask_last_k(nn.Module):
    def __init__(self, hidden_size, k):
        super(Seq_mask_last_k, self).__init__()
        self.k = k
        self.hidden_size = hidden_size
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear_k.weight)

    def forward(self, input):
        """
        :param input: B x L x H
        :return: B x H
        """
        if self.k != 1:
            seq_rep = self.linear_k(torch.mean(input[:, -self.k:, :], dim=1))
        else:
            seq_rep = self.linear_k(input[:, -1, :])
        return seq_rep


class Seq_mask_kth(nn.Module):
    def __init__(self, hidden_size, k):
        super(Seq_mask_kth, self).__init__()
        self.k = k
        self.hidden_size = hidden_size
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear_k.weight)

    def forward(self, input):
        """
        :param input: B x L x H
        :return: B x H
        """
        if self.k != 1:
            seq_rep = self.linear_k(input[:, -self.k, :])
        else:
            seq_rep = self.linear_k(input[:, -1, :])
        return seq_rep


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


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Trend_interest_transforemer_block(nn.Module):
    def __init__(self, args):
        super(Trend_interest_transforemer_block, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, device, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.device = device
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # self.conv3 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # self.chomp3 = Chomp1d(padding)
        # self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(dropout)
        # self.conv4 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # self.chomp4 = Chomp1d(padding)
        # self.relu4 = nn.ReLU()
        # self.dropout4 = nn.Dropout(dropout)
        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                         self.conv2, self.chomp2, self.relu2, self.dropout2).to(self.device)
                                 
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2,
                                 # self.conv3, self.chomp3, self.relu3, self.dropout3,
                                 # self.conv4, self.chomp4, self.relu4, self.dropout4
                                 ).to(self.device)
        self.layer_norm = LayerNorm(n_inputs).to(self.device )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Trend_interest_TemporalConvNet(nn.Module):
    def __init__(self, args):
        super(Trend_interest_TemporalConvNet, self).__init__()
        layers = []
        num_inputs = args.hidden_size
        num_channels = [args.hidden_size] * args.num_blocks
        kernel_size = args.kernel_size
        dropout = args.dropout
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, device=args.device, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        # self.gru_layer = nn.GRU(num_inputs, num_inputs, batch_first=True)
        # self.linear_res = nn.Linear(num_inputs, num_inputs)
        
    def forward(self, x):
        out = self.network(x)
        # out = x+out
        # out, hn = self.gru_layer(out.transpose(1,2))
        # out = out.transpose(1, 2)
        return (x+out), None
        # return out, None
        # return (x+out), hn.squeeze(0)


class Prediction_inner(nn.Module):
    def __init__(self):
        super(Prediction_inner, self).__init__()

    def forward(self, seq_rep, embs):
        return torch.matmul(seq_rep, embs.transpose(0, 1))
        # return F.softmax(torch.matmul(seq_rep, embs.transpose(0, 1)), dim=-1)


class Prediction_linear(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Prediction_linear, self).__init__()
        self.predict_layer = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.predict_layer.weight)

    def forward(self, seq_rep):
        return self.predict_layer(seq_rep)
        # return F.softmax(self.predict_layer(seq_rep), dim=-1)


class MLP_diversity_rep(nn.Module):
    def __init__(self, hidden_size):
        super(MLP_diversity_rep, self).__init__()
        self.linear_w_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_3 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_4 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, inputs):
        out = torch.relu(self.linear_w_1(inputs))
        out = torch.relu(self.linear_w_2(out))
        out = torch.relu(self.linear_w_3(out))
        out = torch.relu(self.linear_w_4(out))
        return out + inputs


class Soft_attention_diversity(nn.Module):
    def __init__(self, hidden_size, device):
        super(Soft_attention_diversity, self).__init__()
        self.linear_w_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_alpha = nn.Linear(hidden_size, 1)
        self.linear_w_2 = nn.Linear(hidden_size, hidden_size)
        self.init_weights()
        self.device = device
        
    def init_weights(self):
        nn.init.xavier_normal_(self.linear_w_1.weight)
        nn.init.xavier_normal_(self.linear_w_2.weight)
        nn.init.xavier_normal_(self.linear_alpha.weight)

    def forward(self, rep_inputs, masks):
        rep_inputs = torch.tanh(self.linear_w_1(rep_inputs))
        alpha_score = self.linear_alpha(rep_inputs).squeeze(-1)

        alpha_score_mask = F.softmax(torch.where(masks == torch.tensor(0).float().to(self.device), torch.tensor(-1e9).float().to(self.device), alpha_score.float()), dim=-1)
        seq_rep = torch.matmul(alpha_score_mask.unsqueeze(1), rep_inputs).squeeze(1)
        return seq_rep


class Linear_Max_pooling_all_diversity(nn.Module):
    def __init__(self, max_len, hidden_size, num_items):
        super(Linear_Max_pooling_all_diversity, self).__init__()
        self.linear_w = nn.Linear(hidden_size, hidden_size)
        self.prediction_layer = nn.Linear(hidden_size, num_items)
        self.max_pooling = nn.MaxPool1d(max_len)

    def forward(self, reps, masks):
        scores = torch.sigmoid(self.prediction_layer(self.linear_w(reps))) * masks
        score = self.max_pooling(scores.transpose(1, 2)).squeeze(-1)
        return score


class Linear_diversity(nn.Module):
    def __init__(self, hidden_size, num_items):
        super(Linear_diversity, self).__init__()
        self.linear_w = nn.Linear(hidden_size, hidden_size)
        self.prediction_layer = nn.Linear(hidden_size, num_items)

    def forward(self, reps):
        score = torch.sigmoid(self.prediction_layer(self.linear_w(reps)))
        return score
