import torch
from torch import nn
from transformers import BertModel
from utils.config import *


class Bert(torch.nn.Module):
    def __init__(self, config: TrainConfig):
        super(Bert, self).__init__()
        # self.bert = BertModel.from_pretrained(config.pretrain_model_path, ignore_mismatched_sizes=True)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.relu = torch.nn.ReLU()
        self.label2idx = config.label_dict
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def forward(self, X):
        context = X[0]
        mask = X[1]
        hidden_layer, pooled = self.bert(input_ids=context, attention_mask=mask, return_dict=False)
        out = self.fc(pooled)
        return self.relu(out)

    def set_bert(self, bert_pretrain):
        self.bert = bert_pretrain

    def predict(self, X, topK):  # this is used only by batch_size = 1
        with torch.no_grad():
            predict = self.forward(X)
            predict_topK = torch.topk(predict, len(self.idx2label), dim=1)[1][0][:topK].tolist()
            predict_topK_label = [self.idx2label[idx] for idx in predict_topK]
            return predict_topK_label


class BertDotAttention(torch.nn.Module):
    def __init__(self, config: TrainConfig):
        super(BertDotAttention, self).__init__()
        self.label2idx = config.label_dict
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        self.bert = BertModel.from_pretrained(config.pretrain_model_path, ignore_mismatched_sizes=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.relu = torch.nn.ReLU()

        in_dim = config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, X):
        context = X[0]
        mask = X[1]
        last_hidden_state, pooled = self.bert(input_ids=context, attention_mask=mask, return_dict=False)

        w = self.attention(last_hidden_state).float()
        w[mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)

        out = self.fc(attention_embeddings)
        return self.relu(out)
