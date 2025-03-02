from embedding import *
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from moe import *

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        residual = x
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask==0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class transformer_block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        print('attn_drop: {}, drop: {}, drop path rate: {}'.format(attn_drop, drop, drop_path))
        self.out_dim = out_dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0.2, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        size = x.shape
        x, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = self.drop_path(self.mlp(self.norm2(x)))

        return x.mean(dim=1).view(size[0], 1, 1, self.out_dim)
    
    

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()
    
    def forward(self, h, t, r, pos_num, norm_transfer):
        # TransD
        h_transfer, r_transfer, t_transfer = norm_transfer
        h = h + torch.sum(h * h_transfer, -1, True) * r_transfer
        t = t + torch.sum(t * t_transfer, -1, True) * r_transfer
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score



class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.batch_size = parameter['batch_size']
        self.embedding = Embedding(dataset, parameter)
        
        if parameter['dataset'] == 'Wiki-One':
            # self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
            #                                             num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
            # self.relation_learner = transformer_block(dim=100, out_dim=50, num_heads=1, drop=0.2, drop_path=0.2)
            self.relation_learner = DeepSeekMoE(input_dim=100, num_experts=32, output_dim=50, k=5)
        elif parameter['dataset'] == 'NELL-One':
            # self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        # num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
            # self.relation_learner = transformer_block(dim=200, out_dim=100, num_heads=1, drop=0.2, drop_path=0.2)
            self.relation_learner = DeepSeekMoE(input_dim=200, num_experts=32, output_dim=100, k=5)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.d_norm_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        
        batch_size = support.shape[0]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        # rel = self.relation_learner(support) # MLP
        rel = self.relation_learner(support.contiguous().view(batch_size, few, -1))  # transformer & MoE
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)
        
        # init a new adaptor
        h_transfer = torch.empty(self.batch_size, 1, 1, self.embed_dim, requires_grad=True).to(self.device)
        r_transfer = torch.empty(self.batch_size, 1, 1, self.embed_dim, requires_grad=True).to(self.device)
        t_transfer = torch.empty(self.batch_size, 1, 1, self.embed_dim, requires_grad=True).to(self.device)
        nn.init.xavier_uniform_(h_transfer)
        nn.init.xavier_uniform_(r_transfer)
        nn.init.xavier_uniform_(t_transfer)
        norm_transfer = (h_transfer, r_transfer, t_transfer)
        
        h_transfer.retain_grad()
        r_transfer.retain_grad()
        t_transfer.retain_grad()

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
            norm_transfer = self.d_norm_sharing[curr_rel]
        else:
            # split on e1/e2 and concat on pos/neg
            sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
            
            '''
            # repeat version
            num_updates = 3  
            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few, norm_transfer)
            y = torch.ones(p_score.shape[0], 1).to(self.device)
            for i in range(num_updates):
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                rel_q = rel - self.beta * rel.grad  
                
                norm_h = h_transfer - self.beta * h_transfer.grad
                norm_r = r_transfer - self.beta * r_transfer.grad
                norm_t = t_transfer - self.beta * t_transfer.grad
                norm_transfer = (norm_h, norm_r, norm_t)
                
                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_q, few, norm_transfer)

            '''
            # no repeat version
            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few, norm_transfer)
            y = torch.ones(p_score.shape[0], 1).to(self.device)
            self.zero_grad()
            loss = self.loss_func(p_score, n_score, y)
            loss.backward(retain_graph=True)

            rel_q = rel - self.beta * rel.grad
            norm_h = h_transfer - self.beta * h_transfer.grad
            norm_r = r_transfer - self.beta * r_transfer.grad
            norm_t = t_transfer - self.beta * t_transfer.grad
            norm_transfer = (norm_h, norm_r, norm_t)
            
            self.d_norm_sharing[curr_rel] = (norm_h.mean(0).unsqueeze(0), 
                                             norm_r.mean(0).unsqueeze(0), 
                                             norm_t.mean(0).unsqueeze(0))
            self.rel_q_sharing[curr_rel] = rel_q # both will be cleared before eval, only val/test uses 
        
        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        if iseval:
            norm_transfer = self.d_norm_sharing[curr_rel]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, norm_transfer)

        return p_score, n_score

