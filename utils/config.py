import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class BasicConfig:

    def __init__(self, dataset_name):
        # data_root_path = os.getcwd() + '/../' + dataset_name
        self.data_root_path = dataset_name
        self.train_path = self.data_root_path + '/raw_train.txt'
        self.train1_path = self.data_root_path + '/raw_train1.txt'
        self.dev_path = self.data_root_path + '/raw_dev.txt'
        self.test_path = self.data_root_path + '/raw_test.txt'
        # self.class_path = self.data_root_path + '/class_F01.txt'
        self.pretrain_model_path = 'pretrain_models/bert_wwm'
        self.save_model_path = ''
        self.log_path = ''

        self.class_path = 'data_module/class.txt'
        self.labels, self.label_dict = self.load_classes(self.class_path)
        self.num_classes = len(self.labels)
        self.label_len = 3

        self.padding_size = 64
        self.hidden_size = 768

        self.rnn_hidden, self.num_layers = 256, 2
        self.num_filters, self.filter_sizes = 256, (3, 4, 5)

        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.parallel = False
        self.save_model = False

    def reset_device(self, new_device):
        self.device = new_device

    def reset_pretrain_path(self, new_pretrain_path):
        self.pretrain_model_path = new_pretrain_path

    def reset_save_model_path(self, new_save_model_path):
        self.save_model_path = new_save_model_path

    def reset_padding_size(self, new_padding_size):
        self.padding_size = new_padding_size

    def reset_classpath(self, class_path):
        self.class_path = class_path
        self.labels, self.label_dict = self.load_classes(self.class_path)
        self.num_classes = len(self.labels)

    @staticmethod
    def load_classes(class_path):
        with open(class_path, mode='r', encoding='utf-8') as file:
            lines = file.readlines()
            labels = [L.strip() for L in lines]
            label_dict = {}
            for index, label in enumerate(labels):
                label_dict[label] = index
        return labels, label_dict


class TrainConfig(BasicConfig):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.epochs = 4
        self.batch_size = 32
        self.lr = 1e-5
        self.dropout = 0.1

    def reset_lr(self, new_lr):
        self.lr = new_lr


class TestConfig(BasicConfig):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.batch_size = 1


# config.reset_model_path('../bert-base-uncased')
# print(config.model_path)
# print(config.test_path)
# print(os.getcwd())

