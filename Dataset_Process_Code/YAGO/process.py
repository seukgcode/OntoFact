import re
def 缩小数据量():
    ontology = {}
    ontology_sx = {}
    triple = set()
    import numpy as np
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/final_triple_ontology.txt') as f:
        for i in f.readlines():
            h,r,t,ho,ro,to = i.strip().split('\t')
            if (ho,ro,to) not in ontology:
                ontology[(ho,ro,to)] = [(h,r,t)]
            else:
                ontology[(ho,ro,to)].append((h,r,t))
    print(len(ontology))
    for i in ontology:
        current_onotlogy = np.random.choice(99, 50, replace=False)
        # print(current_onotlogy)
        # print(ontology[i])
        ontology_sx[i] = np.array(ontology[i])[current_onotlogy]
        for j in ontology_sx[i]:
            triple.add((j[0],j[1],j[2]))
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/数据量缩小/final_triple_ontology.txt','w+') as f:
        for i in ontology_sx:
            for j in ontology_sx[i]:
                f.write(j[0] + '\t' + j[1] + '\t' + j[2] + '\t' + i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
    print(len(triple))

def 缩小数据量构造问答():
    triple = {}
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/数据量缩小/final_triple_ontology.txt') as f:
        for i in f.readlines():
            # print(i)
            h,r,t,ho,ro,to = i.strip().split('\t')
            if (h,r,t) not in triple:
                triple[(h,r,t)] = ''
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/yago_triple_answer.txt') as f:
        for i in f.readlines():
            h,r,t,q,a = i.strip().split('\t')
            if (h,r,t) in triple:
                triple[(h,r,t)] = q
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/数据量缩小/final_triple_question.txt','w+') as f:
        for i in triple:
            f.write(i[0] + '\t' + i[1] + '\t' + i[2] +  '\t' + triple[i] + '\n')

def 统计缩小数据量():
    ontology = set()
    concept = set()
    triple=set()
    entity=set()
    relation = set()
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/数据量缩小/final_triple_ontology.txt') as f:
        for i in f.readlines():
            h,r,t,ho,ro,to = i.strip().split('\t')
            ontology.add((ho,ro,to))
            concept.add(ho)
            concept.add(to)
            entity.add(h)
            entity.add(t)
            triple.add((h,r,t))
            relation.add(r)
    print("本体、概念、三元组、实体、关系", len(ontology), len(concept), len(triple), len(entity), len(relation))

def 抽取chatgpt对应缩小数据量的结果():
    triple = {}
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/数据量缩小/final_triple_ontology.txt') as f:
        for i in f.readlines():
            h,r,t,ho,ro,to = i.strip().split('\t')
            triple[(h,r,t)] = 0
    final_result = open('/data01/c_x/all_result/yago/chatgpt-yago.txt','w+')
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/yago_triple_answer.txt') as f:
        for i in f.readlines():
            h,r,t,q,a = i.strip().split('\t')
            if (h,r,t) in triple:
                final_result.write(i)
def 构造非事实的问题():
    a = open('/data01/c_x/all_result/dbpedia/data/final_triple_NA_question.txt','w+')
    with open('/data01/c_x/all_result/dbpedia/data/final_triple_question.txt') as f:
        for i in f.readlines():
            h,r,t,q = i.strip().split('\t')
            st = q.split(' ')[0]
            a.write('\t'.join([h,r,t,st+' the ' + r + ' of ' + h.replace('_',' ') + ' N/A?'])+'\n')
# 缩小数据量()
# 缩小数据量构造问答()
# 统计缩小数据量()
# 抽取chatgpt对应缩小数据量的结果()\
构造非事实的问题()
