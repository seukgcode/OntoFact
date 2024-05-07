
from SPARQLWrapper import SPARQLWrapper,JSON, CSV
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import re
from itertools import permutations
import csv
import random
from io import StringIO
import os
import shutil
def process_rdf_star():
    new_file = open('/data01/c_x/ChatGPT_Demo/yago/yago-4.5-data/yago-meta-new.ttl','w+')
    with open('/data01/c_x/ChatGPT_Demo/yago/yago-4.5-data/yago-meta.ttl','r',encoding='utf-8') as f:
        for i in f.readlines():
            # print(i)
            i = i.strip().strip('.')
            new_file.write(i+'.'+'\n')
    new_file.close()

def check_data():
    ontology_number = {}
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/all_triple_ht_ontology_data.csv', 'r') as fin:
        reader = csv.reader(fin)
        for index, read in enumerate(reader):
            if read[2] == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                continue
            if read[4] == 'http://www.w3.org/2000/01/rdf-schema#Class' or read[5] == 'http://www.w3.org/2000/01/rdf-schema#Class':
                continue
            if read[5] == 'http://www.w3.org/ns/shacl#NodeShape':
                continue
            if (read[4],read[2],read[5]) not in ontology_number:
                ontology_number[(read[4], read[2], read[5])] = 1
            else:
                ontology_number[(read[4], read[2], read[5])] += 1
            # print(read)
            if index % 1000000 == 0:
                print(index)
    my_df = pd.DataFrame(ontology_number,index=[0]).T.to_csv('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/ontology_number_relation.csv')        

def new_schema_class():
    ontology = pd.read_csv('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/schema_class.csv').to_numpy()[:,0]
    pattern = r"Q\d+"
    cnt = 0
    for i in ontology:
        if re.search(pattern, i):
            continue
        else:
            cnt+=1
    print(cnt)


def process_ontology():
    all_result = []
    valid_ontology = []
    sparql = SPARQLWrapper("http://localhost:3030/yago-4.5-wiki/sparql")
    ontology = pd.read_csv('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/wiki_schema_class.csv').to_numpy()[:,0]
    for i in tqdm(ontology):
        # print(i)
        h = i.strip()
        sparql.setQuery(""" 
        PREFIX schema: <http://schema.org/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        prefix yago: <http://yago-knowledge.org/resource/>
        SELECT ?h ?r ?t (<"""+h+"""> AS ?ho) ?to
        WHERE 
    {
        {
            ?h ?r ?t.
            ?h rdf:type <"""+h+""">.
            ?t rdf:type ?to.
            filter (!regex(str(?to),'Q\\\d+'))
            filter (!regex(str(?to),'http://www.w3.org/2000/01/rdf-schema#Class'))
        }
    }
                    """)
        sparql.setReturnFormat(CSV)
        results = sparql.query().convert()
        # print(type(results))
        pd.read_csv(StringIO(results.decode())).to_csv('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/all_triple_ht_ontology_data.csv',mode='a+',header=False)

def extract_triple():
    ontology = pd.read_excel('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/large_100_yago_wiki_ontology.xlsx').to_numpy()
    fhyq_ontology = {}
    fhqy_triple = []
    final_triple = []
    for i in ontology:
        fhyq_ontology[(i[0],i[1],i[2])] = 0
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/all_triple_ht_ontology_data.csv', 'r') as fin:
        reader = csv.reader(fin)
        for index, read in enumerate(reader):
            if read[2] == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                continue
            if read[4] == 'http://www.w3.org/2000/01/rdf-schema#Class' or read[5] == 'http://www.w3.org/2000/01/rdf-schema#Class':
                continue
            if read[5] == 'http://www.w3.org/ns/shacl#NodeShape':
                continue
            if (read[4],read[2],read[5]) in fhyq_ontology:
                fhqy_triple.append((read[1],read[2],read[3],read[4],read[2],read[5]))
            if index % 1000000 == 0:
                print(index)
    fhqy_triple = list(set(fhqy_triple))
    random.shuffle(fhqy_triple)
    for i in fhqy_triple:
        if (i[3],i[4],i[5]) in fhyq_ontology and fhyq_ontology[(i[3],i[4],i[5])]<100:
            fhyq_ontology[(i[3],i[4],i[5])] += 1
            final_triple.append(list(i))
    my_df = pd.DataFrame(final_triple).to_csv('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/sample_final_triple.csv')

def check_triple_data():
    ontology_triple = {}
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/sample_final_triple.csv', 'r') as fin:
        reader = csv.reader(fin)
        for index, i in enumerate(reader):
            if index == 0:
                continue
            if (i[4],i[5],i[6]) not in ontology_triple:
                ontology_triple[(i[4],i[5],i[6])] = 1
            else:
                ontology_triple[(i[4],i[5],i[6])]+=1
    for i in ontology_triple:
        assert ontology_triple[i] == 100, print(ontology_triple[i])
    print(len(ontology_triple))

def generate_unique_triple():
    triple = []
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/sample_final_triple.csv', 'r') as fin:
        reader = csv.reader(fin)
        for index, i in enumerate(reader):
            if index == 0:
                continue
            triple.append((i[1],i[2],i[3]))
    triple = list(set(triple))
    triple = [list(i) for i in triple]
    pd.DataFrame(triple).to_csv('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/final_triple.csv')

def check_unique_triple():
    cnt = 0
    entity = set()
    relation=set()
    # yago_triple = open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/yago_triple.txt', 'w+')
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/final_triple.csv', 'r') as fin:
        reader = csv.reader(fin)
        for index, i in enumerate(reader):
            if index == 0:
                continue
            cnt += 1
            entity.add(i[1])
            entity.add(i[3])
            relation.add(i[2])
            # yago_triple.write(i[1].split('/')[-1] + '\t' + i[2].split('/')[-1] + '\t' + i[3].split('/')[-1] + '\n')
    # yago_triple.close()
    print('triple number:',cnt)
    print('entity number:',len(entity))
    print('relation number:',len(relation))

def rule_unicode(name):
    if '_u0026_' in name:
        name = name.replace('_u0026_','&')
    if '_u0028_' in name:
        name = name.replace('_u0028_','(')
    if '_u0029_' in name:
        name = name.replace('_u0029_',')')
    if '_u002C_' in name:
        name = name.replace('_u002C_',',')
    if '_u002E_' in name:
        name = name.replace('_u002E_','.')
    if '_u0027_' in name:
        name = name.replace('_u0027_','\'')
    if '_u0022_' in name:
        name = name.replace('_u0022_','"')
    if '_u0021_' in name:
        name = name.replace('_u0021_','!')
    if '_u002F_' in name:
        name = name.replace('_u002F_','/')
    if '_u002B_' in name:
        name = name.replace('_u002B_','+')
    if '_u0024_' in name:
        name = name.replace('_u0024_','$')
    if '_u002A_' in name:
        name = name.replace('_u002A_','*')
    if '_u0025_' in name:
        name = name.replace('_u0025_','%')
    
    return name
    


def check_unicode_triple_txt():
    check_file = open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/check_unicode_yago_triple.txt','w+')
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/yago_triple.txt') as f:
        for i in f.readlines():
            h,r,t = i.strip().split()
            h = rule_unicode(h)
            t = rule_unicode(t)
            if 'u002' in h or 'u002' in t or 'u002' in r:
                print(i)
                exit(0)
            check_file.write(h+'\t'+r+'\t'+t+'\n')
    
    check_file.close()

def split_yago_dataset():

    import math
    all_triple = []
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/check_unicode_yago_triple.txt') as f:
        for i in f.readlines():
            h,r,t = i.strip().split('\t')
            all_triple.append((h,r,t))
    max_num = math.ceil(len(all_triple) / 15000.0)

    for i in range(max_num):
        start = i * 15000
        end = min(len(all_triple), (i + 1) * 15000)
        with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/split/yago_wiki_4.5_{}.txt'.format(i), 'w+') as f:
            for j in all_triple[start:end]:
                f.write(j[0]+'\t'+j[1]+'\t'+j[2]+'\n')

def move_result_file():
    for root, files, file in os.walk('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/xx_result'):
        for i in file:
            with open(os.path.join(root,i)) as f:
                data = f.readlines()
                if i != 'yago_wiki_4.5_86_result.txt':
                    assert len(data) == 15000, print(os.path.join(root,i))
                # all_the_text = open(file_path).read()
                for l in range(len(data)):
                    try:
                        h,r,t,q,a = data[l].strip().split('\t')
                    except:
                        print(os.path.join(root,i), l,data[l])
                        exit()
                # for ii in data:
                #     ii = ii.strip()
                #     if len(ii) == 0:
                #         continue
                #     with open(os.path.join('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/x_result',i), 'a+') as f:
                #         f.write(ii+'\n')

def move_result_file1():
    d = []
    for root, files, file in os.walk('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/xx_result'):
        for i in sorted(file,key=lambda x:int(x.split('_')[-2])):
            with open(os.path.join(root,i)) as f:
                for j in f.readlines():
                    h,r,t,q,a = j.strip().split('\t')

                    d.append(h + '\t' + r + '\t' + t + '\t' + q + '\t' + a + '\n')
        with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/yago_triple_answer.txt', 'w+') as f:
            f.writelines(d)

            # text = open(os.path.join(root,i)).read()
            # text = re.sub(r'\nNo\.\n', '\tNo.\n',text)
            # text = re.sub(r'\nYes\.\n', '\tYes.\n',text)
            # text = re.sub(r'\nNo\n', '\tNo.\n',text)
            # text = re.sub(r'\nYes\n', '\tYes.\n',text)
            # with open(os.path.join('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/xx_result',i), 'w+') as f:
            #         f.write(text)
            # print(text)

def generate_unique_triple_ontology():
    triple = []
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/sample_final_triple.csv', 'r') as fin:
        reader = csv.reader(fin)
        for index, i in enumerate(reader):
            if index == 0:
                continue
            triple.append(rule_unicode(i[1].split('/')[-1])+ '\t' +rule_unicode(i[2].split('/')[-1])+ '\t' +rule_unicode(i[3].split('/')[-1])+ '\t' +i[4].split('/')[-1]+ '\t' +rule_unicode(i[5].split('/')[-1])+ '\t' +i[6].split('/')[-1]+ '\n')
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/final_triple_ontology.txt', 'w+') as fin:
        fin.writelines(triple)

def tj():
    concept = set()
    with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/sample_final_triple.csv', 'r') as fin:
        reader = csv.reader(fin)
        for index, i in enumerate(reader):
            if index == 0:
                continue
            concept.add(i[6])
            concept.add(i[4])
            
    print(len(concept))

def chinese_tj():
    concept = set()
    with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/allfieldtriple.ttl', 'r') as fin:
        reader = csv.reader(fin)
        for index, i in enumerate(reader):
            i = i[0].strip(' .').replace('> <','\t').replace('> "','\t').replace(' ','_').replace('<','').replace('>','')
            print(i)
            if index==100:
                exit()
            
    

# process_ontology()
# new_schema_class()
# check_data()
# extract_triple()
# check_triple_data()
# generate_unique_triple()
# check_unique_triple()
# check_unicode_triple_txt()
# split_yago_dataset()
# move_result_file()
# move_result_file1()
# generate_unique_triple_ontology()
# tj()
chinese_tj()