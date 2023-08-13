from flask import Flask, request, jsonify
from annoy import AnnoyIndex
from transformers import BertTokenizer, BertModel
import pickle, os, pymysql
from utils.config import *
from utils.data_utils import *


# load AnnoyIndex from ann file
def load_annoy():
    dict = {}
    with open('data_module/class.txt') as file:
        lines = file.readlines()
        codes = [line.strip() for line in lines]
        for code in codes:
            # annoy索引文件和索引-标题映射
            file1, file2 = 'data_module/%s.ann' % code, 'data_module/%s.pkl' % code,
            if not os.path.exists(file1) or not os.path.exists(file2):
                continue
            T = AnnoyIndex(768, 'angular')
            T.load(file1)
            with open(file2, 'rb') as f:
                index_to_title = pickle.load(f)
            dict[code] = [T, index_to_title]
            print('load {} finish'.format(code))
    return dict


# use Bert do embed
def embed_document(doc):
    inputs = bert_tokenizer(doc[0], doc[1], doc[2],
                            return_tensors="pt",
                            max_length=512,
                            truncation=True,
                            padding="max_length")
    outputs, pooled = bert_model(input_ids=inputs['input_ids'].to(config.device),
                                 attention_mask=inputs['attention_mask'].to(config.device),
                                 return_dict=False)
    return inputs, pooled.cpu().detach().numpy()[0]


# do top 10 similar search return titles
def similar_search(data):
    _, doc_embedding = embed_document([data['title'], data['keywords'], data['abstract']])
    code = data['code']
    T, idx2title = annoy_tables.get(code)
    indexes = T.get_nns_by_vector(doc_embedding, 10)
    titles = [idx2title[i] for i in indexes]
    return titles


def build_paper_dict():
    title_to_paper = {}
    with open('data_module/class.txt') as file:
        lines = file.readlines()
        codes = [line.strip() for line in lines]
        for code in codes:
            # 论文数据文件
            path = 'data_module/{}_data.txt'.format(code)
            if os.path.exists(path):
                with open(path, encoding='utf-8', mode='r') as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines]
                    records = [line.split('\t') for line in lines]
                    for r in records:
                        title_to_paper[r[0]] = r
    return title_to_paper


def get_by_title(title):
    sql = 'select pbi.title, pbi.keywords, pdi.abstract \
        from project_basic_info pbi left join project_detail_info pdi \
        on pbi.query_code=pdi.query_code \
        where pbi.title="{}"'.format(title)
    cursor = db.cursor()
    cursor.execute(sql)
    rtn = cursor.fetchone()
    fields = list(rtn)
    if rtn is None:
        return None
    else:
        fields = list(rtn)
    return {'title': fields[0], 'keywords': fields[1], 'abstract': fields[2]}


def get_by_title2(title):
    fields = title_to_paper.get(title)
    if fields is None:
        return None
    return {'title': fields[0], 'keywords': fields[1], 'abstract': fields[2]}


def build_graph(papers):
    titles = []
    nodes = []
    links = []
    eles = set()
    for p in papers:
        titles.append(p['title'])
        eles.add(p['title'])
        for w in p['keywords'].split(';'):
            eles.add(w)
    ele_index = {}
    # 添加节点
    for index, ele in enumerate(eles):
        ele_index[ele] = index
        # 区分标题、关键词节点
        type = 'T' if ele in titles else 'K'
        nodes.append({'id':index, 'name':ele, 'type': type})
    link_index = 0
    # 添加关系
    for p in papers:
        t = p['title']
        for w in p['keywords'].split(';'):
            links.append({'id':link_index, 'from_id':ele_index.get(t), 'to_id':ele_index.get(w)})
            link_index = link_index+1
    return {'nodes': nodes, 'links':links}


def build_word_cloud(papers):
    keywords_list = [p['keywords'] for p in papers]
    word_freq = {}
    for keywords in keywords_list:
        for w in keywords.split(';'):
            if word_freq.get(w) is None:
                word_freq[w] = 1
            else:
                word_freq[w] = word_freq[w] + 1
    return word_freq


config = TestConfig(dataset_name='None')
config.device = 'cpu'
config.label_len = 5
config.pretrain_model_path = 'bert_wwm'
config.save_model_path = 'Bert_wwm2.pt'
config.reset_classpath('data_module/class.txt')
with open('data_module/class2name.pkl', 'rb') as file:
    code2name = pickle.load(file)

app = Flask('国自然申请代码智能推荐')
annoy_tables = load_annoy()
bert_tokenizer = BertTokenizer.from_pretrained('./bert_wwm')
bert_model = BertModel.from_pretrained('./bert_wwm')
bert_model.to(device=config.device)
predict_model = load_model(config, 'Bert', bert_model)
title_to_paper = build_paper_dict()

# db = pymysql.connect(host='localhost',
#                      user='root',
#                      password='123456',
#                      database='nature_foundation')


@app.route('/api/test')
def test():
    return 'Hello, Flask!'


@app.route('/api/recommend', methods=['POST'])
def predict():
    data = request.json
    title, keywords, abstract = data['title'], data['keywords'], data['abstract']
    top_k = data['top_k']
    inputs, _ = embed_document([title, keywords, abstract])
    top_5_code = predict_model.predict(X=[inputs['input_ids'], inputs['attention_mask']], topK=top_k)
    top_5_name = [code2name[code] for code in top_5_code]
    return jsonify({'codes': top_5_code, 'names': top_5_name })


@app.route('/api/deep-mining', methods=['POST'])
def deep_mining():
    data = request.json
    similar_titles = similar_search(data)
    papers = [get_by_title2(t) for t in similar_titles]
    papers = [p for p in papers if p is not None]
    return jsonify({'papers': papers,
                    'word_cloud': build_word_cloud(papers),
                    'knowledge_graph': build_graph(papers)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
