from transformers import BertModel
from torch import nn
from utils.config import *


class  Bert_CNN(nn.Module):

    def __init__(self, config: TrainConfig):
        super(Bert_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.label2idx = config.label_dict
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    @staticmethod
    def conv_and_pool(x, conv):
        x = nn.functional.relu(conv(x)).squeeze(3)  # (1,256,31,1) -> (1,256,31)
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)  # (1,256,1) -> (1,256)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        out = encoder_out.unsqueeze(1)  # output (1, 1, 32, 768)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # output (1, 768)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out

    def predict(self, X, topK):  # this is used only by batch_size = 1
        with torch.no_grad():
            predict = self.forward(X)
            predict_topK = torch.topk(predict, len(self.idx2label), dim=1)[1][0][:topK].tolist()
            predict_topK_label = [self.idx2label[idx] for idx in predict_topK]
            return predict_topK_label