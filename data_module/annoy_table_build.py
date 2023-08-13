import numpy as np
import pymysql
from transformers import BertTokenizer, BertModel
from annoy import AnnoyIndex
import os
import pickle


tokenizer = BertTokenizer.from_pretrained('../bert_wwm')
model = BertModel.from_pretrained('../bert_wwm')


def build_annoy_index(embeddings, n_trees=10):
    f = embeddings[0].shape[0]  # Embedding dimension
    t = AnnoyIndex(f, 'angular')
    for i, emb in enumerate(embeddings):
        t.add_item(i, emb)
    t.build(n_trees)
    return t


def find_similar_documents(new_doc, t, n=5):
    new_doc_embedding = embed_document(new_doc)
    return t.get_nns_by_vector(new_doc_embedding, n)


def embed_document(doc):
    inputs = tokenizer(doc[0], doc[1], doc[2],
                       return_tensors="pt", max_length=512,
                       truncation=True, padding="max_length")
    outputs, pooled = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=False)
    return pooled.detach().numpy()[0]


def convert_db_data(_records):
    lines = []
    for _record in _records:
        fields = list(_record)
        fields = [f.strip() for f in fields]  # 数据清洗，洗掉空格，换行符，关键词清洗，()
        fields = [f.replace("“", "\"") for f in fields]
        fields = [f.replace("”", "\"") for f in fields]
        fields = [f.replace("\n", "") for f in fields]
        fields = [f.replace("\r", "") for f in fields]
        fields = [f.replace("\t", "") for f in fields]
        lines.append(fields)
    return lines


def get_records(code):
    path = code + '_data.txt'
    if os.path.exists(path):
        with open(path, encoding='utf-8', mode='r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            records = [line.split('\t') for line in lines]
            return records
    else:
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='123456',
                             database='nature_foundation')
        sql = 'select pbi.title, pbi.keywords, pdi.abstract \
                    from project_basic_info pbi left join project_detail_info pdi \
                    on pbi.query_code=pdi.query_code \
                    where pbi.category like "%s" and pbi.permission_year < 2018' % code
        cursor = db.cursor()
        cursor.execute(sql)
        records = convert_db_data(cursor.fetchall())
        with open(path, encoding='utf-8', mode='w') as file:
            lines = ['\t'.join(record) for record in records]
            file.write('\n'.join(lines))
        return records


def build_annoy_file(code):
    records = get_records(code)
    if len(records) == 0:
        return
    # 索引文件
    if os.path.exists(code + '.ann'):
        T = AnnoyIndex(768, 'angular')
        T.load(code + '.ann')
    else:
        embeddings = np.load(code + '.npy')
        T = build_annoy_index(embeddings, 10)
        T.save(code + '.ann')
    # 字典文件
    if os.path.exists(code + '.pkl'):
        with open(code + '.pkl', 'rb') as f:
            index_to_title = pickle.load(f)
    else:
        index2title = {i: record[0] for i, record in enumerate(records)}
        with open(code + '.pkl', 'wb') as f:
            pickle.dump(index2title, f)


# def test():
#     #有限长空间耦合LDPC码的设计与译码研究[SEP]空间耦合;译码算法;速率兼容;LDPC码[SEP]LDPC码作为下一代移动通信（5G）宽带业务数据信息的长码块编码方案，具有逼近容量限的译码性能和并行译码等优点，可保障数据的高速可靠传输。作为LDPC码新发现的SC-LDPC码，其所具有的低复杂度结构特性和阈值饱和特性，为LDPC码的研究提供了新方法。本项目拟针对有限长SC-LDPC码，设计一套实用化的码构造方法和译码算法。首先，利用SC-LDPC码的阈值饱和特性，给出一种实现复杂度低、可逼近容量限的速率兼容SC-LDPC码的设计方法。然后，针对传统滑窗译码算法的性能损失问题，基于深度学习方法，设计一种低时延低复杂度高可靠性的滑窗译码算法。最后，针对所设计的译码算法，研究有限长译码性能分析方法，用于预估有限长SC-LDPC码在该译码算法下的译码性能。通过本项目的研究，在完善有限长SC-LDPC码理论研究的同时为5G系统中LDPC码的设计提供理论依据和技术支撑。	F0101
#     doc = ['有限长空间耦合LDPC码的设计与译码研究', '空间耦合;译码算法;速率兼容;LDPC码', 'LDPC码作为下一代移动通信（5G）宽带业务数据信息的长码块编码方案，具有逼近容量限的译码性能和并行译码等优点，可保障数据的高速可靠传输。作为LDPC码新发现的SC-LDPC码，其所具有的低复杂度结构特性和阈值饱和特性，为LDPC码的研究提供了新方法。本项目拟针对有限长SC-LDPC码，设计一套实用化的码构造方法和译码算法。首先，利用SC-LDPC码的阈值饱和特性，给出一种实现复杂度低、可逼近容量限的速率兼容SC-LDPC码的设计方法。然后，针对传统滑窗译码算法的性能损失问题，基于深度学习方法，设计一种低时延低复杂度高可靠性的滑窗译码算法。最后，针对所设计的译码算法，研究有限长译码性能分析方法，用于预估有限长SC-LDPC码在该译码算法下的译码性能。通过本项目的研究，在完善有限长SC-LDPC码理论研究的同时为5G系统中LDPC码的设计提供理论依据和技术支撑。	F0101']
#     # embedding = embed_document(doc)
#     indexes = find_similar_documents(doc, T, 5)
#     for index in indexes:
#         print(index2title[index])

# build ann file
with open('class.txt') as file:
    lines = file.readlines()
    codes = [line.strip() for line in lines]
    for code in codes:
        print(code)
        build_annoy_file(code)
