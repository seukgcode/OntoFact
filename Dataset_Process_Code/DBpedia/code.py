'''
|Author: RainbowCatSzy
|Date: 2023-06-25 20:44:48
|LastEditors: RainbowCatSzy
|LastEditTime: 2023-06-25 21:04:21
'''

import numpy as np
def bc_triple_tj():
    ontology = {}
    with open('/data/c_x/bc_triple.txt') as f:
        for i in f.readlines():
            try:
                h,r,t,ho,ro,to = i.strip().split('\t')
            except:
                continue
            if (ho,ro,to) not in ontology:
                ontology[(ho,ro,to)]=[(h,r,t)]
            else:
                ontology[(ho,ro,to)].append((h,r,t))
    with open('/data/c_x/unique_triple/bc_triple_tj.txt','w+') as f:
        for i in ontology:
            f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + str(len(set(ontology[i]))) + '\n')

def tj_mean_show_time_triple_ontology():
    triple = {}
    with open('/data/c_x/depedia_final_data/final_triple_ontology.txt') as f:
        for i in f.readlines():
            h,r,t,ho,ro,to = i.strip().split('\t')
            if (h,r,t)not in triple:
                triple[(h,r,t)]=[(ho,ro,to)]
            else:
                triple[(h,r,t)].append((ho,ro,to))
    num = []
    print(len(triple))
    for i in triple:
        num.append(len(triple[i]))
    print(num)
    print(np.mean(num))

if __name__ == '__main__':
    # bc_triple_tj()
    tj_mean_show_time_triple_ontology()