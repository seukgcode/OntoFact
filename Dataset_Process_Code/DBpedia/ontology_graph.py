'''
|Author: RainbowCatSzy
|Date: 2023-06-02 15:36:43
|LastEditors: RainbowCatSzy
|LastEditTime: 2023-06-02 16:31:36
'''

# ontology_graph = []
# with open('/data/c_x/depedia_final_data/all_ontology_tj_05281647.txt') as f:
#     for i in f.readlines():
#         h,r,t,n = i.strip().split('\t')
#         ontology_graph.append((h,r,t))
# ontology_graph = set(ontology_graph)
# with open('/data/c_x/depedia_final_data/representation_learning_data/ontology_graph.txt', 'w+') as f:
#     f.writelines([i[0] + '\t' + i[1] + '\t' + i[2] + '\n' for i in ontology_graph])
ontology_graph = []
entity = []
with open('/data/c_x/depedia_final_data/final_triple_ontology.txt') as f:
    for i in f.readlines():
        h,r,t,ho,ro,to = i.strip().split('\t')
        ontology_graph.append((h,ho))
        ontology_graph.append((t,to))
        entity.append(h)
        entity.append(t)
ontology_graph = set(ontology_graph)

with open('/data/c_x/depedia_final_data/representation_learning_data/entity_type_ontology.txt', 'w+') as f:
    for i in ontology_graph:
        f.write(i[0] + '\t' + 'type' + '\t' + i[1] + '\n')


# # ontology_graph = []
# entity = []
# with open('/data/c_x/depedia_final_data/chatgpt_final_result/all_triple_chatgpt_result.txt') as f:
#     for i in f.readlines():
#         h,r,t,_,_ = i.strip().split('\t')
#         # ontology_graph.append((h,ho))
#         # ontology_graph.append((t,to))
#         entity.append(h)
#         entity.append(t)
# print(len(list(set(entity))))
# # print(len(list(set(ontology_graph))))