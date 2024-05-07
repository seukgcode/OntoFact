'''
|Author: RainbowCatSzy
|Date: 2023-05-01 20:06:51
|LastEditors: RainbowCatSzy
|LastEditTime: 2023-06-01 12:54:06
'''
from SPARQLWrapper import SPARQLWrapper,JSON
from tqdm import trange
import numpy as np
import os
import random
sparql = SPARQLWrapper("http://localhost:8890/sparql/")
random.seed(242)
np.random.seed(242)
def extract():
    entity_list=[]
    ontology_list=[]
    with open('/data/c_x/current_ontology_number_xy_100.txt','r') as f:
        for line in f.readlines():
            h, r, t, num = line.strip().split('\t')
            # data=eval(data)
            ontology_list.append((h, r, t))
            
    # print(ontology_list)
    bc_triple = open('/data/c_x/bc_triple_new_0528_new.txt','w+')
    for i in trange(len(ontology_list)):
        ontology=ontology_list[i]
    # for ontology in ontology_list:
        h_ontology = ontology[0]
        r_ontology = ontology[1]
        t_ontology = ontology[2]
        # h_ontology = 'Work'
        # r_ontology = 'founder'
        # t_ontology = 'Animal'
        # h_ontology = 'Animal'
        # r_ontology = 'prospectTeam'
        # t_ontology = 'Agent'
        sparql.setQuery("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX dbr: <http://dbpedia.org/resource/>
                PREFIX dbo: <http://dbpedia.org/ontology/>
                PREFIX dbp: <http://dbpedia.org/property/>
                
                select *
                    from <https://dbpedia.org>
                    where{
                        {?s <http://dbpedia.org/ontology/"""+ r_ontology +"""> ?o.
                        ?s rdf:type <http://dbpedia.org/ontology/"""+ h_ontology +""">.
                        ?o rdf:type <http://dbpedia.org/ontology/"""+ t_ontology +""">.
                        }
                        UNION
                        {
                        ?s <http://dbpedia.org/property/"""+ r_ontology +"""> ?o.
                        ?s rdf:type <http://dbpedia.org/ontology/"""+ h_ontology +""">.
                        ?o rdf:type <http://dbpedia.org/ontology/"""+ t_ontology +""">.
                        }
                    }
            """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()['results']['bindings']
        if results != []:
            if len(results) < 100:
                continue
            triple = set()
            # new_result = random.sample(results,100)
            for res in results:
                s=res['s']['value'].split('/')[-1]
                o=res['o']['value'].split('/')[-1]
                if s == '' or o == '':
                    continue
                triple.add((s,r_ontology,o))
            if len(triple) < 90:
                continue
            elif len(triple) > 100:
                new_result = random.sample(triple,100)
            
            for i in new_result:
                bc_triple.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t'+ h_ontology +'\t' + r_ontology + '\t' + t_ontology +'\n')        


def process():
    triple = set()
    with open('/data/c_x/bc_triple.txt') as file:
        lines = file.readlines()
        for l in lines:
            if len(l.strip().split('\t')) != 6:
                continue
            h,r,t,ho,_,to = l.strip().split('\t')
            triple.add((h,r,t,ho,to))
    triple_dict = {}
    for (h,r,t,ho,to) in triple:
        if (ho,r,to) not in triple_dict:
            triple_dict[(ho,r,to)] = [(h,r,t)]
        else:
            if len(triple_dict[(ho,r,to)]) < 100:
                triple_dict[(ho,r,to)].append((h,r,t))
    
    with open('/data/c_x/bc_triple_new_tj.txt', 'w+') as file:
        for i in triple_dict:
            file.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + str(len(triple_dict[i])) + '\n')

    with open('/data/c_x/bc_triple_new.txt', 'w+') as file:
        for i in triple_dict:
            for j in triple_dict[i]:
                file.write(j[0] + '\t' + j[1] + '\t' + j[2] + '\t' + i[0] + '\t' + i[1] + '\t' + i[2] + '\n')

def tj():
    ontology_list = []
    with open('/data/c_x/xy_10.txt','r') as f:
        for line in f.readlines():
            data = line.strip()
            data=eval(data)
            ontology_list.append(data)
    number_ontology = {}
    cnt = 0
    with open('/data/c_x/bc_triple_new_tj.txt','r') as f:
        for line in f.readlines():
            h,r,t,n = line.strip().split('\t')
            number_ontology[(h, r, t)] = n
            if int(n)<10:
                cnt+=1
    print(cnt)
def cl():
    current_ontology = open('./current_ontology_number_xy_100_0528.txt', 'w+')
    ontology_schema_mapping = np.load('/data/c_x/ChatGPT_Demo/new/dataset/all_entity_0_demo/data/ontology_schema_mapping.npy', allow_pickle=True).item()
    for i in ontology_schema_mapping:
        ontology_schema_mapping[i] = np.unique(ontology_schema_mapping[i], axis=0)
        if len(ontology_schema_mapping[i])>100:
                index = np.random.choice(len(ontology_schema_mapping[i]), 100)
                ontology_schema_mapping[i] = np.array(ontology_schema_mapping[i])[index]
    all_triple = set()
    for i in ontology_schema_mapping:
        if len(ontology_schema_mapping[i])==100:
            continue
        current_ontology.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + str(len(ontology_schema_mapping[i])) + '\n')
        for ii in ontology_schema_mapping[i]:
            all_triple.add((ii[0],ii[1],ii[2]))
    current_ontology.close()
    print(len(all_triple))

def u_n():
    triple = set()
    with open('/data/c_x/bc_triple_new_0528.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            if len(l.strip().split('\t')) != 6:
                continue
            h,r,t,ho,_,to = l.strip().split('\t')
            triple.add((h,r,t,ho,to))
    with open('/data/c_x/bc_triple_new_0528_ready.txt', 'w+') as file:
        for i in triple:
            file.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\t' + i[1] + '\t' + i[4] + '\n')

def hb():
    ontology_schema_mapping = np.load('/data/c_x/ChatGPT_Demo/new/dataset/all_entity_0_demo/data/ontology_schema_mapping.npy', allow_pickle=True).item()
    for i in ontology_schema_mapping:
        ontology_schema_mapping[i] = [(i[0], i[1], i[2]) for i in ontology_schema_mapping[i]]
        ontology_schema_mapping[i] = list(set(ontology_schema_mapping[i]))
        if len(ontology_schema_mapping[i])>100:
            ontology_schema_mapping[i] = random.sample(ontology_schema_mapping[i],100)
    all_triple = set()
    for i in ontology_schema_mapping:
        if len(ontology_schema_mapping[i]) != 100:
            continue
        for ii in ontology_schema_mapping[i]:
            all_triple.add((ii[0],ii[1],ii[2],i[0],i[1],i[2]))
    # current_ontology.close()
    with open('/data/c_x/dbpedia_bc_final_entity_ontology.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            h,r,t, ho, ro, to = l.strip().split('\t')
            all_triple.add((h,r,t,ho,ro,to))
    
    with open('/data/c_x/depedia_final_data/dbpedia_bc_final_entity_ontology_05281643.txt', 'w+') as file:
        for i in all_triple:
            file.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\t' + i[4] + '\t' + i[5] + '\n')
    print(len(all_triple))

def relation_tj():
    relation = set()
    with open('/data/c_x/bc_triple_new_0528_all.txt','r') as f:
        for line in f.readlines():
            h,r,t = line.strip().split('\t')
            relation.add(r)
    print(len(relation))

def new_tj():
    traiple = set()
    with open('/data/c_x/train_new_result.txt','r') as f:
        for line in f.readlines():
            h,r,t,q,a = line.strip().split('\t')
            traiple.add((h,r,t))

    relation = set()
    with open('/data/c_x/depedia_final_data/dbpedia_bc_final_triple.txt','r') as f:
        for line in f.readlines():
            h,r,t = line.strip().split('\t')
            relation.add((h,r,t))
    has_triple = set()
    for i in traiple:
        if i in relation:
            has_triple.add(i)
            relation.discard(i)
    with open('/data/c_x/depedia_final_data/triple_for_chatgpt_sy.txt','w+') as f:
        for i in has_triple:
            f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
    with open('/data/c_x/depedia_final_data/triple_for_chatgpt_all.txt','w+') as f:
        for i in relation:
            f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')

def new_tj_ontology():
    ontology_triple = dict()
    triple = set()
    with open('/data/c_x/depedia_final_data/final_triple_ontology.txt', 'r') as f:
        for i in f.readlines():
            h,r,t,ho,ro,to = i.strip().split('\t')
            triple.add((h,r,t))
            if (ho,ro,to) not in ontology_triple:
                ontology_triple[(ho,ro,to)] = [(h,r,t)]
            else:
                ontology_triple[(ho,ro,to)].append((h,r,t))
    with open('/data/c_x/depedia_final_data/dbpedia_bc_final_triple.txt', 'w+') as f:
        for i in triple:
            f.write(i[0]+'\t'+i[1]+'\t'+i[2]+'\n')
    with open('/data/c_x/depedia_final_data/all_ontology_tj_05281647.txt', 'w+') as f:
        for i in ontology_triple:
            f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + str(len(set(ontology_triple[i]))) + '\n')


def test_number():
    triple = dict()
    with open('/data/c_x/bc_triple.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            if len(l.strip().split('\t')) != 6:
                continue
            h,r,t,ho,_,to = l.strip().split('\t')
            if (ho,r,to) not in triple:
                triple[(ho,r,to)] = [(h,r,t)]
            else:
                triple[(ho,r,to)].append((h,r,t))
    bc_triple_ready = open('/data/c_x/bc_triple_ontology_0528_ready_05281427.txt', 'w+')
    bc_triple= open('/data/c_x/bc_triple_0528_ready_05281427.txt', 'w+')
    tripp = set()
    with open('/data/c_x/bc_triple_0528_tj.txt', 'w+') as f:
        for i in triple:
            if len(set(triple[i])) > 100:
                f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + str(len(set(triple[i]))) + '\n')
                new_triple = random.sample(set(triple[i]), 100)
                for j in new_triple:
                    bc_triple_ready.write(j[0] + '\t' + j[1] + '\t' + j[2] + '\t' + i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
                    tripp.add((j[0],j[1],j[2]))
    bc_triple_ready.close()
    for i in tripp:
        bc_triple.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
    bc_triple.close()


def tt():
    ontology_schema_mapping = np.load('/data/c_x/ChatGPT_Demo/new/dataset/all_entity_0_demo/data/ontology_schema_mapping.npy', allow_pickle=True).item()
    ff = open('/data/c_x/depedia_final_data/final_tj.txt','w+')
    for i in ontology_schema_mapping:
        ontology_schema_mapping[i] = [(i[0], i[1], i[2]) for i in ontology_schema_mapping[i]]
        ontology_schema_mapping[i] = list(set(ontology_schema_mapping[i]))
        if len(ontology_schema_mapping[i]) < 100:
            ff.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')

def tt1():
    ontology_schema_mapping = np.load('/data/c_x/ChatGPT_Demo/new/dataset/all_entity_0_demo/data/ontology_schema_mapping.npy', allow_pickle=True).item()
    ff = open('/data/c_x/depedia_final_data/final_tj_large_100.txt','w+')
    for i in ontology_schema_mapping:
        ontology_schema_mapping[i] = [(i[0], i[1], i[2]) for i in ontology_schema_mapping[i]]
        ontology_schema_mapping[i] = list(set(ontology_schema_mapping[i]))
        if len(ontology_schema_mapping[i]) > 100:
            ff.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + str(len(ontology_schema_mapping[i])) + '\n')

def ttt():
    triple = dict()
    with open('/data/c_x/bc_triple.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            if len(l.strip().split('\t')) != 6:
                continue
            h,r,t,ho,_,to = l.strip().split('\t')
            if (ho,r,to) not in triple:
                triple[(ho,r,to)] = [[h,r,t]]
            else:
                triple[(ho,r,to)].append([h,r,t])
    
    bc_triple_ready = open('/data/c_x/depedia_final_data/bc_triple_ontology_0528_ready_05281427.txt', 'w+')
    bc_triple= open('/data/c_x/depedia_final_data/bc_triple_0528_ready_05281427.txt', 'w+')
    
    ontology_schema_mapping = np.load('/data/c_x/ChatGPT_Demo/new/dataset/all_entity_0_demo/data/ontology_schema_mapping.npy', allow_pickle=True).item()
    tripp = []
    for i in ontology_schema_mapping:
        if len(ontology_schema_mapping[i])<100:
            if (i[0], i[1], i[2]) not in triple:
                continue
            all_t = np.append(np.array(ontology_schema_mapping[i]),np.array(triple[(i[0], i[1], i[2])]))
            all_t = np.unique(all_t, axis=0)
            if len(all_t)>100:
                index = np.random.choice(len(all_t), 100)
                all_t = all_t[index]
            for j in all_t:
                bc_triple_ready.write(j[0] + '\t' + j[1] + '\t' + j[2] + '\t' + i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
                tripp.append((j[0],j[1],j[2]))
    bc_triple_ready.close()
    tripp = set(tripp)
    for i in tripp:
        bc_triple.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
    bc_triple.close()

def tttttt():
    ontology_schema_mapping = np.load('/data/c_x/ChatGPT_Demo/new/dataset/all_entity_0_demo/data/ontology_schema_mapping.npy', allow_pickle=True).item()
    ff = open('/data/c_x/bc_ontology.txt', 'w+')
    for i in ontology_schema_mapping:
        temp = np.unique(np.array(ontology_schema_mapping[i]),axis = 0)
        if len(temp)<100:
            ff.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
    ff.close()


def final_triple():
    triple = dict()
    with open('/data/c_x/bc_triple.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            if len(l.strip().split('\t')) != 6:
                continue
            h,r,t,ho,_,to = l.strip().split('\t')
            if (ho,r,to) not in triple:
                triple[(ho,r,to)] = [[h,r,t]]
            else:
                triple[(ho,r,to)].append([h,r,t])


def sssplit():
    relation = set()
    with open('/data/c_x/depedia_final_data/triple_for_chatgpt.txt','r') as f:
        for line in f.readlines():
            h,r,t = line.strip().split('\t')
            relation.add((h,r,t))
    relation = list(relation)

def finallll():
    sy = set()
    with open('/data/c_x/depedia_final_data/triple_for_chatgpt_sy.txt','r') as f:
        for line in f.readlines():
            h,r,t = line.strip().split('\t')
            sy.add((h,r,t))
    sy_result = open('/data/c_x/syyyyyyyyy.txt','w+')
    with open('/data/c_x/train_new_result.txt','r') as f:
        for line in f.readlines():
            h,r,t,q,a = line.strip().split('\t')
            if (h,r,t) in sy:
                sy_result.write(h + '\t' + r + '\t' + t + '\t' + q + '\t' + a + '\n')

    
if __name__ == '__main__':
    # process()
    # cl()
    # extract()
    # u_n()
    # hb()
    # relation_tj()
    # new_tj()
    finallll()
    # new_tj_ontology()
    # test_number()
    # tt1()
    # tttttt()

    # a = [(1,2,3),(4,5,6),(7,8,9)]
    # print(random.sample(a,2))

