
dd = open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/english/20230714140951_english_only_triple.txt')
dd1 = []
for i in dd.readlines():
    h,r,t = i.strip().split('\t')
    dd1.append([h,t])
dd2 = []
with open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/english/20230714140951_english_only_triple_one.txt') as f:
    for i in f.readlines():
        h,r,t = i.strip().split('\t')
        dd2.append([h,t])
diction = {}
for i in range(len(dd1)):
    h,t = dd1[i]
    h_1,t_1 = dd2[i]
    if h not in diction:
        diction[h] = h_1
    elif h_1 != diction[h]:
        print(h_1,diction[h])
        exit()
    if t not in diction:
        diction[t] = t_1
    elif t_1 != diction[t]:
        print(t_1,diction[t])
        exit()

a = open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/english/20230714140951_english_triple_ontology_final.txt','w+')
with open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/english/20230714140951_english_triple_ontology.txt') as f:
    for i in f.readlines():
        h,r,t,ho,ro,to = i.strip().split('\t')
        h = diction[h]
        t = diction[t]
        a.write('\t'.join([h,r,t,ho,ro,to])+'\n')
