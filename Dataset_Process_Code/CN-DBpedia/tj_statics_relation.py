import csv
import pandas as pd
import numpy as np
triple_ontology = {}
concept = set()
property = set()
wlxs_entity = {}
triple = {}
ontology_num = {}

with open('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/qc_wlxs_final_num.csv', 'r', encoding='utf8') as fin:
  reader = csv.reader(fin)
  for index, read in enumerate(reader):
      if index == 0:
         continue
      triple_ontology[(read[0],read[1],read[2])] = int(read[3])
triple_ontology = sorted(triple_ontology.items(), key = lambda kv:(kv[1]),reverse=True)
triple_ontology = {i[0]:i[1] for i in triple_ontology[:17000]}
skip_e = {'网络小说', '虚拟人物'}
with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/7Lore_all_triple_ontology.csv', 'r', encoding='utf8') as fin:
  reader = csv.reader(fin)
  for index, read in enumerate(reader):
      if index % 1000000 == 0:
             print(index)
      
      if read[-2] in skip_e:
         wlxs_entity[read[1]] = 0
      
      if read[-1] in skip_e:
         wlxs_entity[read[3]] = 0
      
      if read[2] == '连载平台':
         wlxs_entity[read[1]] = 0
         wlxs_entity[read[3]] = 0

# entity = set()
# relation = set()
# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/ffinal_7Lore_all_triple_ontology.txt','r') as f:
#    for i in f.readlines():
#       h,r,t = i.strip().split('\t')
#       entity.add(h)
#       entity.add(t)
#       relation.add(r)
# print(len(entity))
# print(len(relation))      

# print(len(concept))
# print(len(property))
# exit()

triple = {}
ontology_num = {}
with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/7Lore_all_triple_ontology.csv', 'r', encoding='utf8') as fin:
  reader = csv.reader(fin)
  for index, read in enumerate(reader):
    if index % 1000000 == 0:
        print(index)
    if read[2] in {'连载平台','中文名','中文名称', '连载状态'} or read[1] == read[3] or read[1] in wlxs_entity or read[3] in wlxs_entity:
         continue
   #  if (read[-2],read[2],read[-1]) not in ontology_num:
   #     ontology_num[(read[-2],read[2],read[-1])] = set()
   #     ontology_num[(read[-2],read[2],read[-1])].add((read[1],read[2],read[3]))
   #  else:
   #     ontology_num[(read[-2],read[2],read[-1])].add((read[1],read[2],read[3]))
    # if index == 41000000:
    #        break
    if (read[-2],read[2],read[-1]) not in triple_ontology:
      continue
    elif (read[-2],read[2],read[-1]) in triple:
       triple[(read[-2],read[2],read[-1])].add((read[1],read[2],read[3]))
    else:
       triple[(read[-2],read[2],read[-1])] = set()
       triple[(read[-2],read[2],read[-1])].add((read[1],read[2],read[3]))
# num = {}
# for i in ontology_num:
#    length = len((ontology_num[i]))
#    num[i] = length

# my_df = pd.DataFrame(num,index=[0]).T.to_csv('/data01/c_x/ChatGPT_Demo/yago/precoess-yago-4.5/final_num.csv')        
    
final_triple = set()
with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/20230713154311_ffinal_7Lore_all_triple_ontology.txt','w+') as f:
   for i in triple:
      length = len(triple[i])
      current_idx = np.random.choice(length, 50, replace=False)
      current_triple = np.array(list(triple[i]))[current_idx]
      for ii in current_triple:
         f.write(ii[0] + '\t' + ii[1] + '\t' + ii[2] + '\t' + i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
         final_triple.add((ii[0],ii[1],ii[2]))
with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/20230713154311_only_triple.txt','w+') as f:
   for i in final_triple:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')

# # my_df = pd.DataFrame(triple_ontology,index=[0]).T.to_csv('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/7lore_tj.csv')