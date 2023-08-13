from transformers import BertModel
from torch import nn
from utils.config import *


class Bert_RCNN(torch.nn.Module):
    def __init__(self, config: TrainConfig):
        super(Bert_RCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size,
                            config.rnn_hidden,
                            config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.max_pool = nn.MaxPool1d(config.padding_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)
        self.label2idx = config.label_dict
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, pooled = self.bert(input_ids=context, attention_mask=mask, return_dict=False)  # batch padding 768
        out, _ = self.lstm(encoder_out)  # batch padding 512
        out = torch.cat((encoder_out, out), 2)  # batch padding 1280
        out = torch.nn.functional.relu(out)
        out = out.permute(0, 2, 1)  # batch 1280 padding
        out = self.max_pool(out).squeeze()  # batch 1280
        out = self.fc(out)  # batch 6(num_classes)
        return out

    def predict(self, X, topK):  # this is used only by batch_size = 1
        with torch.no_grad():
            predict = self.forward(X)
            predict = torch.unsqueeze(predict, dim=0)
            predict_topK = torch.topk(predict, len(self.idx2label), dim=1)[1][0][:topK].tolist()
            predict_topK_label = [self.idx2label[idx] for idx in predict_topK]
            return predict_topK_label
