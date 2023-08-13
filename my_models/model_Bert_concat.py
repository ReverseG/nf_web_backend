import torch.nn
from torch import nn
from transformers import BertModel
from utils.config import *


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class Bert_Concat1(torch.nn.Module):
    def __init__(self, config: TrainConfig):
        super(Bert_Concat1, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        self.bert2 = BertModel.from_pretrained(config.pretrain_model_path)

        self.layer_norm1 = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = torch.nn.LayerNorm(config.hidden_size)

        for param in self.bert.parameters():
            param.requires_grad = True

        # self.attention = torch.nn.MultiheadAttention(embed_dim=config.hidden_size*2, num_heads=2)
        # self.attention = Attention(config.hidden_size, 3)
        # self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)

        self.fc = torch.nn.Linear(config.hidden_size * 2, config.num_classes)

        self.relu = torch.nn.ReLU()

    def forward(self, X1, X2):
        context1 = X1[0]
        mask1 = X1[1]
        _, pooled1 = self.bert(input_ids=context1, attention_mask=mask1, return_dict=False)

        context2 = X2[0]
        mask2 = X2[1]
        _, pooled2 = self.bert2(input_ids=context2, attention_mask=mask2, return_dict=False)

        # context3 = X3[0]
        # mask3 = X3[1]
        # _, pooled3 = self.bert3(input_ids=context3, attention_mask=mask3, return_dict=False)

        # attention_weights = self.attention(pooled).squeeze(dim=-1)
        # attention_scores = torch.softmax(attention_weights, dim=-1)
        # context_vector = torch.matmul(attention_scores.unsqueeze(dim=1), pooled).squeeze(dim=1)
        # attention_context, attention_weight = self.attention(pooled, pooled, pooled)

        # pooled1, pooled2, pooled3 = torch.unsqueeze(pooled1, dim=1), \
        #                             torch.unsqueeze(pooled2, dim=1), \
        #                             torch.unsqueeze(pooled3, dim=1)
        # pooled = torch.concat([pooled1, pooled2, pooled3], dim=1)  # batch, 1536
        # attention_context = self.attention(pooled)

        pooled = torch.concat([self.layer_norm1(pooled1), self.layer_norm2(pooled2)], dim=1)  # batch, 1536
        # pooled = torch.concat([pooled1, pooled2], dim=1)  # batch, 1536

        out = self.fc(pooled)
        return self.relu(out)


class Bert_Concat_Try(torch.nn.Module):
    def __init__(self, config: TrainConfig):
        super(Bert_Concat_Try, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        self.bert2 = BertModel.from_pretrained(config.pretrain_model_path)

        self.layer_norm1 = torch.nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = torch.nn.LayerNorm(config.hidden_size)

        for param in self.bert.parameters():
            param.requires_grad = True
        # 1.1 convolution
        self.conv = nn.Conv2d(1, config.num_filters, (3, config.hidden_size))
        # 1.2
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])

        # 2. attention
        in_dim = config.hidden_size
        self.attention1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

        # self.attention2 = nn.Sequential(
        #     nn.Linear(in_dim, in_dim),
        #     nn.LayerNorm(in_dim),
        #     nn.GELU(),
        #     nn.Linear(in_dim, 1),
        # )

        self.euclidean = torch.nn.PairwiseDistance(p=2)

        # 1. convolution
        # self.fc = torch.nn.Linear(config.hidden_size + config.num_filters, config.num_classes)
        # 2. attention
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.num_classes)

        self.relu = torch.nn.ReLU()

    @staticmethod
    def conv_and_pool(x, conv):
        x = x.permute(1, 0, 2)  # 8*10*768
        x = x.unsqueeze(1)
        x = nn.functional.relu(conv(x)).squeeze(3)  # 8, 256, 8
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)  # (1,256,1) -> (1,256)
        return x

    def forward(self, X1, X2s):
        title_hidden_state, title_pooled = self.bert(input_ids=X1[0], attention_mask=X1[1], return_dict=False)

        keyword_embeddings = []
        for context, mask in zip(X2s[0], X2s[1]):
            _, pooled2 = self.bert2(input_ids=context, attention_mask=mask, return_dict=False)
            keyword_embeddings.append(pooled2.unsqueeze(0))
        keyword_embedding = torch.concat(keyword_embeddings, dim=0)

        # 1. convolution
        # keyword_embedding = self.conv_and_pool(keyword_embedding, self.conv)
        keyword_embedding = torch.cat([self.conv_and_pool(keyword_embedding, conv) for conv in self.convs], 1)
        # 2. attention
        # keyword_embedding = keyword_embedding.permute(1, 0, 2)
        # w1 = self.attention1(keyword_embedding).float()
        # w1 = torch.softmax(w1, 1)
        # keyword_embedding = torch.sum(w1 * keyword_embedding, dim=1)

        # w2 = self.attention2(title_hidden_state).float()
        # # w[mask == 0] = float('-inf')
        # w2 = torch.softmax(w2, 1)
        # title_embedding = torch.sum(w2 * title_hidden_state, dim=1)

        pooled = torch.concat([title_pooled, keyword_embedding], dim=1)
        out = self.fc(pooled)

        # eu_distance = torch.sum(self.euclidean(title_pooled, keyword_embedding)) / 1000
        eu_distance = None

        return self.relu(out), eu_distance


class Bert_TKA(torch.nn.Module):
    def __init__(self, config: TrainConfig):
        super(Bert_TKA, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        self.use_attention = False
        self.bert2 = BertModel.from_pretrained(config.pretrain_model_path)
        self.bert3 = BertModel.from_pretrained(config.pretrain_model_path)

        for param in self.bert.parameters():
            param.requires_grad = True

        in_dim = config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )
        self.fc1 = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.fc2 = torch.nn.Linear(config.hidden_size * 3, config.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, X1, X2, X3):
        context1 = X1[0]
        mask1 = X1[1]
        _, pooled1 = self.bert(input_ids=context1, attention_mask=mask1, return_dict=False)
        context2 = X2[0]
        mask2 = X2[1]
        _, pooled2 = self.bert2(input_ids=context2, attention_mask=mask2, return_dict=False)
        context3 = X3[0]
        mask3 = X3[1]
        _, pooled3 = self.bert3(input_ids=context3, attention_mask=mask3, return_dict=False)

        if self.use_attention:
            pooled1, pooled2, pooled3 = torch.unsqueeze(pooled1, dim=1), \
                                        torch.unsqueeze(pooled2, dim=1), \
                                        torch.unsqueeze(pooled3, dim=1)
            pooled = torch.concat([pooled1, pooled2, pooled3], dim=1)  # batch, 3, 768
            w = self.attention(pooled).float()
            w = torch.softmax(w, 1)
            attention_context = torch.sum(w * pooled, dim=1)
            out = self.fc1(attention_context)
        else:
            pooled = torch.concat([pooled1, pooled2, pooled3], dim=1)
            out = self.fc2(pooled)

        return self.relu(out)

    def open_attention(self):
        self.use_attention = True


import torch
import torch.nn as nn


class Multi_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Multi_Attention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)

        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        att, p = self.attention(q, k, v)

        att = self.out(att)

        return att


class Bert_TKA_MultiHead(torch.nn.Module):
    def __init__(self, config: TrainConfig):
        super(Bert_TKA_MultiHead, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        self.use_attention = False

        for param in self.bert.parameters():
            param.requires_grad = True

        self.attention = Multi_Attention(config.hidden_size, 1)
        self.fc = torch.nn.Linear(config.hidden_size * 3, config.num_classes)

        self.relu = torch.nn.ReLU()

    def forward(self, X1, X2, X3):
        context1 = X1[0]
        mask1 = X1[1]
        _, pooled1 = self.bert(input_ids=context1, attention_mask=mask1, return_dict=False)
        context2 = X2[0]
        mask2 = X2[1]
        _, pooled2 = self.bert(input_ids=context2, attention_mask=mask2, return_dict=False)
        context3 = X3[0]
        mask3 = X3[1]
        _, pooled3 = self.bert(input_ids=context3, attention_mask=mask3, return_dict=False)
        pooled1, pooled2, pooled3 = torch.unsqueeze(pooled1, dim=1), \
                                    torch.unsqueeze(pooled2, dim=1), \
                                    torch.unsqueeze(pooled3, dim=1)
        pooled = torch.concat([pooled1, pooled2, pooled3], dim=1)  # batch, 3, 768

        att = self.attention(pooled)
        att = att.view(att.shape[0], -1)
        out = self.fc(att)

        return self.relu(out)

    def open_attention(self):
        self.use_attention = True


class Bert_TKA32(torch.nn.Module):
    """
    对摘要内容进行切分后，做句子级别的embedding，然后经过卷积提取特征信息，然后与标题和关键词的特征做拼接
    """

    def __init__(self, config: TrainConfig):
        super(Bert_TKA32, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        in_dim = config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )
        self.fc = torch.nn.Linear(config.hidden_size * 3, config.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, X1, X2, X3s):
        context1 = X1[0]
        mask1 = X1[1]
        _, pooled1 = self.bert(input_ids=context1, attention_mask=mask1, return_dict=False)

        context2 = X2[0]
        mask2 = X2[1]
        _, pooled2 = self.bert(input_ids=context2, attention_mask=mask2, return_dict=False)

        # context3 = X3[0]
        # mask3 = X3[1]
        # _, pooled3 = self.bert(input_ids=context3, attention_mask=mask3, return_dict=False)
        abstract_embeddings = []
        for context, mask in zip(X3s[0], X3s[1]):
            _, pooled2 = self.bert(input_ids=context, attention_mask=mask, return_dict=False)
            abstract_embeddings.append(pooled2.unsqueeze(0))
        abstract_embeddings = torch.concat(abstract_embeddings, dim=0)
        abstract_embeddings = abstract_embeddings.permute(1, 0, 2)
        w1 = self.attention(abstract_embeddings).float()
        w1 = torch.softmax(w1, 1)
        pooled3 = torch.sum(w1 * abstract_embeddings, dim=1)

        pooled = torch.concat([pooled1, pooled2, pooled3], dim=1)  # batch, 1536

        out = self.fc(pooled)
        return self.relu(out)
