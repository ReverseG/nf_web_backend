from transformers import BertTokenizer, BertModel
import numpy as np


device='cuda:2'
tokenizer = BertTokenizer.from_pretrained('../bert_wwm')
model = BertModel.from_pretrained('../bert_wwm')
model.to(device=device)


def embed_document(doc):
    inputs = tokenizer(doc[0], doc[1], doc[2],
                       return_tensors="pt",
                       max_length=512,
                       truncation=True,
                       padding="max_length")
    outputs, pooled = model(input_ids=inputs['input_ids'].to(device),
                            attention_mask=inputs['attention_mask'].to(device),
                            return_dict=False)
    return pooled.cpu().detach().numpy()[0]


def get_records(code):
    path = code + '_data.txt'
    with open(path, encoding='utf-8', mode='r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        records = [line.split('\t') for line in lines]
    return records


with open('class.txt') as file:
    lines = file.readlines()
    codes = [line.strip() for line in lines]
    for code in codes:
        documents = get_records(code)
        embeddings = [embed_document(d) for d in documents]
        embeddings = np.array(embeddings)
        np.save(code + '.npy', embeddings)
