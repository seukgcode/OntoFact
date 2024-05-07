import tqdm
from collections import defaultdict
import numpy as np
ontology_triple = defaultdict(set)
triple_ontology = defaultdict(set)
with open('/data01/c_x/all_result/yago/original_file/final_triple_ontology.txt') as f:
    for i in f.readlines():
        h,r,t,h0,r0,t0 = i.strip().split('\t')
        ontology_triple[(h0,r0,t0)].add((h,r,t))
        triple_ontology[(h,r,t)].add((h0,r0,t0))
triple_new_triple = {}
for i in tqdm.tqdm(triple_ontology):
    # print(i)
    choice = np.random.choice(len(triple_ontology[i]),1)[0]
    # print(list(triple_ontology[i])[choice])
    ho,ro,to = list(triple_ontology[i])[choice]
    tail = set(np.array(list(ontology_triple[(ho,ro,to)]))[:,2]) - {i[2]}
    try:
        triplec_choice = np.random.choice(len(tail),1)[0]
        triple_new_triple[(i[0],i[1],i[2])] = (i[0],i[1],list(tail)[triplec_choice])
        # print(triple_new_triple[(i[0],i[1],i[2])])
        # exit()
    except:
        head = set(np.array(list(ontology_triple[(ho,ro,to)]))[:,0]) - {i[0]}
        triplec_choice = np.random.choice(len(head),1)[0]
        triple_new_triple[(i[0],i[1],i[2])] = (list(head)[triplec_choice],i[1],i[2])

with open('/data01/c_x/all_result/yago/original_file/hard_negative_samples.txt','w+') as f:
    for i in triple_new_triple:
        f.write('\t'.join([i[0],i[1],i[2],triple_new_triple[i][0],triple_new_triple[i][1],triple_new_triple[i][2]])+'\n')