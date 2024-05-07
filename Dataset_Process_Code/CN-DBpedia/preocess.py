import itertools
import numpy as np
import tqdm
# import chardet
# entity = set()
# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/firstreleasetriple.ttl', encoding='GB18030') as f:
#     for i in f.readlines():
#         h,r,t = i.strip().strip('.\n').strip().split('> <')
#         h = h.strip('<').split('/')[-1].replace(' ','')
#         t = t.strip('>').split('/')[-1].replace(' ','')
#         entity.add(h)
# print(len(entity))
# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/entity.txt', 'w+') as f:
#     for i in entity:
#         f.write(i+'\n')

# tt = open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/triple.txt', 'w+')
type_tt = open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/all_entity_type.txt', 'w+')
with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/allfieldtriple.ttl', encoding= 'utf-8') as f:
    data = f.readlines()
    for i in range(len(data))[34070741:]:
        if i % 1000000 == 0:
            print(i)
        type_tt.write(data[i])
        e = data[i].strip().strip('.\n').strip().split('> <')
        if len(e) == 3:
            h,r,t = e
            h = h.strip('<').split('/')[-1]
            t = t.strip('>').split('/')[-1]
            # print(h,r,t)
            # if r == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                # print(i)
                # exit()
                # type_tt.write(h + '\t' + r + '\t' + t + '\n')
            # else:
            #     tt.write(h + '\t' + r + '\t' + t + '\n')
        elif len(e) == 2:
            continue
            h,_ = entity
            h = h.strip('<').split('/')[-1].replace(' ','')
# tt.close()
type_tt.close()

# ontology_entity = {}
# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/entity_type.txt') as f:
#     for i in f.readlines():
#         try:
#             h,r,t = i.strip().split('\t')
#         except:
#             continue
#         if h not in ontology_entity:
#             ontology_entity[h] = [t]
#         else:
#             ontology_entity[h].append(t)

# ontology_num = {}
# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/triple.txt') as f:
#     for i in f.readlines():
#         try:
#             h,r,t = i.strip().split('\t')
#         except:
#             continue
#         r = r.split('/')[-1].replace(' ','')
#         try:
#             ontology_h = set(ontology_entity[h])
#             ontology_t = set(ontology_entity[t])
#         except:
#             continue
#         c = itertools.product(ontology_h,ontology_t)
#         for i in c:
#             if (i[0],r,i[1]) not in ontology_num:
#                 ontology_num[(i[0],r,i[1])] = 1
#             else:
#                 ontology_num[(i[0],r,i[1])] += 1

# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/ontology_num.txt', 'w+') as f:
#     for i in ontology_num:
#         if ontology_num[i] >=100:
#             f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + str(ontology_num[i]) + '\n')
# a = []
# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/ontology_num.txt') as f:
#     for i in f.readlines():
#         h,r,t,num = i.strip().split('\t')
#         a.append([h,r,t,int(num)])
# a = sorted(a,key=lambda x: x[3], reverse=True)
# with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/ontology_num_reverse.txt', 'w+') as f:
#     f.writelines([i[0]+'\t'+i[1]+'\t'+i[2]+'\t'+str(i[3])+'\n'  for i in a])
# # cnt = 0
# # ontology = []
# # with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/ontology_num_reverse.txt') as f:
# #     for i in f.readlines():
# #         h,r,t,num = i.strip().split('\t')
# #         num = int(num)
# #         ontology.append((h,r,t))
# #         cnt+=1
# #         if cnt == 15845:
# #             break

# # triple = {}
# # with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/triple.txt') as f:
# #     for i in tqdm.tqdm(f.readlines()[:5000]):
# #         try:
# #             h,r,t = i.strip().split('\t')
# #         except:
# #             continue
# #         r = r.split('/')[-1].replace(' ','')
# #         if r == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
# #             continue 
# #         try:
# #             ontology_h = set(ontology_entity[h])
# #             ontology_t = set(ontology_entity[t])
# #         except:
# #             continue
# #         c = itertools.product(ontology_h,ontology_t)
# #         for j in c:
# #             if (j[0],r,j[1]) in ontology:
# #                 if (j[0],r,j[1]) not in triple:
# #                     triple[(j[0],r,j[1])] = [(h,r,t)]
# #                 else:
# #                     triple[(j[0],r,j[1])].append((h,r,t))

# # with open('/data01/c_x/ChatGPT_Demo/Chinese/general/pku-pie/sample_triple.txt', 'w+') as f:
# #     for i in triple:
# #         idx = np.random.choice(len(triple[i]), 1, replace=False)
# #         current = np.array(triple[i])[idx]
# #         f.writelines([j[0] + '\t' + j[1] + '\t' + j[2] + '\t' + i[0] + '\t' + i[1] + '\t' + i[2] + '\n' for j in current])



        
        