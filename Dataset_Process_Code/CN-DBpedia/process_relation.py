import sys
import csv
import pandas as pd
import itertools
r = ['小说进度','连载网站', '每集长度', '字级', '类型', '色彩' , '在线播放平台', '房间数量', '页数']
t = ['男','女']
entity = {}
with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/7Lore_entity_ontology.txt', 'r') as fin:
    for i in fin.readlines():
        # print(i)
        h,_,t = i.strip().split('\t')
        if h not in entity:
            entity[h] = [t]
        else:
            entity[h].append(t)
            entity[h] = list(set(entity[h]))

cnt = 0
# new = open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/7Lore_entity_ontology_entity.txt', 'w+')
with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/7Lore_triple.csv', 'r', encoding='utf8') as fin:
  reader = csv.reader(fin)
  for index, read in enumerate(reader):
        h = read[0].strip()
        t = read[2].strip()
        if read[1].strip() == 'tag' or h == t or read[1].strip() in r:
            # new.write(h + '\t' + read[1].strip() + '\t' + t + '\n')
            continue
        if h in entity and t in entity:
            cnt += 1
            h_o = entity[h]
            t_o = entity[t]
            c = itertools.product(h_o,t_o)
            for i in c:
                d = pd.DataFrame([[h,read[1].strip(),t,i[0],i[1]]])
                d.to_csv('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/7Lore_all_triple_ontology.csv', mode='a+', header= False)
        if index % 1000000 == 0:
            print(index,cnt)


