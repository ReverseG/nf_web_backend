from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils.config import *
import re
from my_models.model_BertCNN import Bert_CNN
from my_models.model_BertRCNN import Bert_RCNN
from my_models.model_Bert import Bert


def load_model(config: TrainConfig, model_type, bert_pretrain):
    model = None
    if model_type == 'Bert_RCNN':
        model = Bert_RCNN(config=config)
    elif model_type == 'Bert_CNN':
        model = Bert_CNN(config=config)
    elif model_type == "Bert":
        model = Bert(config=config)
    model.to(device=config.device)
    # model.set_bert(bert_pretrain)
    model.load_state_dict(torch.load(config.save_model_path, map_location=config.device))
    return model


class NFDataset(Dataset):
    def __init__(self, data_type, config: TrainConfig):
        self.texts = []
        self.inputs = []
        self.code_labels = []
        self.index_labels = []
        if data_type == 'train':
            data_file_path = config.train_path
        elif data_type == 'dev':
            data_file_path = config.dev_path
        elif data_type == 'test':
            data_file_path = config.test_path
        else:
            data_file_path = data_type
        tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path, ignore_mismatched_sizes=True)
        with open(data_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text, code_label = line.split('\t')
                text, code_label = text.strip(), code_label.strip()
                # 测试去除 人工智能代码 及 多媒体信息代码后的预测准确率
                # if label in ['F0117', 'F0210'] or label[:3] == 'F06':
                #     continue
                # if code_label is None or len(code_label) != 5:
                #     continue
                index = config.label_dict.get(code_label[:config.label_len])
                if index is None:
                    continue
                input = tokenizer(text, padding='max_length', max_length=config.padding_size,
                                  truncation=True, return_tensors='pt')
                self.texts.append(text)
                self.inputs.append(input)
                self.code_labels.append(code_label)
                self.index_labels.append(index)
        self.index_labels = torch.LongTensor(self.index_labels)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def get_batch_labels(self, idx):
        return self.index_labels[idx].numpy()

    def get_batch_codes(self, idx):
        return self.code_labels[idx]

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def __len__(self):
        return len(self.index_labels)

    def __getitem__(self, idx):
        batch_texts = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


# tile[SEP]keyword1[SEP]keyword2[SEP]abstract
class NFDatasetSep(Dataset):
    def __init__(self, data_type, config: TrainConfig):
        self.texts = []
        self.inputs = []
        self.code_labels = []
        self.index_labels = []
        if data_type == 'train':
            data_file_path = config.train_path
        elif data_type == 'train1':
            data_file_path = config.train1_path
        elif data_type == 'dev':
            data_file_path = config.dev_path
        elif data_type == 'test':
            data_file_path = config.test_path
        else:
            data_file_path = data_type
        tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path, ignore_mismatched_sizes=True)
        with open(data_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text, code_label = line.split('\t')
                text, code_label = text.strip(), code_label.strip()
                # 测试去除 人工智能代码 及 多媒体信息代码后的预测准确率
                # if label in ['F0117', 'F0210'] or label[:3] == 'F06':
                #     continue
                # if code_label is None or len(code_label) != 5:
                #     continue
                title, keywords, abstract = text.split('[SEP]')
                keywords = '[SEP]'.join(keywords.split(';'))
                text = '[SEP]'.join([title, keywords, abstract])
                if config.label_len is not None:
                    code_label = code_label[:config.label_len]
                index = config.label_dict.get(code_label)
                if index is None:
                    continue
                input = tokenizer(text, padding='max_length', max_length=config.padding_size,
                                  truncation=True, return_tensors='pt')
                self.texts.append(text)
                self.inputs.append(input)
                self.code_labels.append(code_label)
                self.index_labels.append(index)
        self.index_labels = torch.LongTensor(self.index_labels)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def get_batch_labels(self, idx):
        return self.index_labels[idx].numpy()

    def get_batch_codes(self, idx):
        return self.code_labels[idx]

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def __len__(self):
        return len(self.index_labels)

    def __getitem__(self, idx):
        batch_texts = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


# T K1 K2 K3 K4 K5 K6
class NFDatasetTitleK6(Dataset):
    def __init__(self, data_type, config: TrainConfig):
        self.texts = []
        self.input1s = []
        self.input2s = []
        self.labels = []
        if data_type == 'train':
            data_file_path = config.train_path
        elif data_type == 'dev':
            data_file_path = config.dev_path
        else:
            data_file_path = config.test_path
        tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
        with open(data_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text, label = line.split('\t')
                text, label = text.strip(), label.strip()
                title, keyword = text.split('[SEP]')
                # title
                input1 = tokenizer(title, padding='max_length', max_length=128,
                                  truncation=True, return_tensors='pt')

                # keyword
                words = split_keyword(keyword)
                if len(words) > 6:
                    words = words[:6]
                else:
                    words = words + ['[PAD]']*(6-len(words))

                input2 = [tokenizer(word, padding='max_length', max_length=12, truncation=True, return_tensors='pt')
                          for word in words]

                if label is None:
                    continue
                label = config.label_dict.get(label[:config.label_len])
                if label is None:
                    continue
                self.texts.append(text)
                self.input1s.append(input1)
                self.input2s.append(input2)
                self.labels.append(label)
        self.labels = torch.LongTensor(self.labels)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def get_batch_input1s(self, idx):
        return self.input1s[idx]

    def get_batch_input2s(self, idx):
        return self.input2s[idx]

    def get_batch_labels(self, idx):
        return self.labels[idx].numpy()

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_text1s = self.get_batch_input1s(idx)
        batch_text2s = self.get_batch_input2s(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_text1s, batch_text2s, batch_y


class NFConcatDatasetTKA(Dataset):
    def __init__(self, data_type, config: TrainConfig):
        self.texts = []
        self.input1s = []
        self.input2s = []
        self.input3s = []
        self.labels = []
        if data_type == 'train':
            data_file_path = config.train_path
        elif data_type == 'dev':
            data_file_path = config.dev_path
        else:
            data_file_path = config.test_path
        tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
        with open(data_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text, label = line.split('\t')
                text, label = text.strip(), label.strip()
                title, keyword, abstract = text.split('[SEP]')
                # title
                input1 = tokenizer(title, padding='max_length', max_length=100,
                                  truncation=True, return_tensors='pt')

                # keyword
                keyword = ';'.join(split_keyword(keyword))
                input2 = tokenizer(keyword, padding='max_length', max_length=100,
                                   truncation=True, return_tensors='pt')

                # abstract
                input3 = tokenizer(abstract, padding='max_length', max_length=512,
                                   truncation=True, return_tensors='pt')
                if label is None:
                    continue
                label = config.label_dict.get(label[:config.label_len])
                if label is None:
                    continue
                self.texts.append(text)
                self.input1s.append(input1)
                self.input2s.append(input2)
                self.input3s.append(input3)
                self.labels.append(label)
        self.labels = torch.LongTensor(self.labels)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def get_batch_input1s(self, idx):
        return self.input1s[idx]

    def get_batch_input2s(self, idx):
        return self.input2s[idx]

    def get_batch_input3s(self, idx):
        return self.input3s[idx]

    def get_batch_labels(self, idx):
        return self.labels[idx].numpy()

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_text1s = self.get_batch_input1s(idx)
        batch_text2s = self.get_batch_input2s(idx)
        batch_text3s = self.get_batch_input3s(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_text1s, batch_text2s, batch_text3s, batch_y


class NFConcatDatasetTKA32(Dataset):
    def __init__(self, data_type, config: TrainConfig):
        self.texts = []
        self.input1s = []
        self.input2s = []
        self.input3s = []
        self.labels = []
        if data_type == 'train':
            data_file_path = config.train_path
        elif data_type == 'dev':
            data_file_path = config.dev_path
        else:
            data_file_path = config.test_path
        tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
        with open(data_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text, label = line.split('\t')
                text, label = text.strip(), label.strip()
                title, keyword, abstract = text.split('[SEP]')
                # title
                input1 = tokenizer(title, padding='max_length', max_length=64,
                                   truncation=True, return_tensors='pt')

                # keyword
                keyword = ';'.join(split_keyword(keyword))
                input2 = tokenizer(keyword, padding='max_length', max_length=64,
                                   truncation=True, return_tensors='pt')

                # abstract
                abstract_segs = re.split('[，。；,.;]', abstract)
                if len(abstract_segs) > 32:
                    abstract_segs = abstract_segs[:32]
                else:
                    abstract_segs = abstract_segs + ['这是一个填充句子。'] * (32 - len(abstract_segs))
                input3 = [tokenizer(a, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
                          for a in abstract_segs]
                if label is None:
                    continue
                label = config.label_dict.get(label[:config.label_len])
                if label is None:
                    continue
                self.texts.append(text)
                self.input1s.append(input1)
                self.input2s.append(input2)
                self.input3s.append(input3)
                self.labels.append(label)
        self.labels = torch.LongTensor(self.labels)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def get_batch_input1s(self, idx):
        return self.input1s[idx]

    def get_batch_input2s(self, idx):
        return self.input2s[idx]

    def get_batch_input3s(self, idx):
        return self.input3s[idx]

    def get_batch_labels(self, idx):
        return self.labels[idx].numpy()

    def get_batch_inputs(self, idx):
        return self.inputs[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_text1s = self.get_batch_input1s(idx)
        batch_text2s = self.get_batch_input2s(idx)
        batch_text3s = self.get_batch_input3s(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_text1s, batch_text2s, batch_text3s, batch_y


def wash(string):
    # 将括号包围的多义词remove掉
    pattern = r'([a-zA-Z）（)(-。="]*)'
    for s in string.split(';'):
        for match in re.findall(pattern, s):
            s = s.replace(match, '')
    return s


def split_keyword(keyword):
    words = keyword.split(';')
    words = [wash(word) for word in words]
    words = [word for word in words if len(word) > 1 and len(word) <= 10]
    return words