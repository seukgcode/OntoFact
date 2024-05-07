'''
|Author: RainbowCatSzy
|Date: 2023-05-25 22:52:14
|LastEditors: RainbowCatSzy
|LastEditTime: 2023-06-24 16:19:21
'''
# has_result_triple = []
# path = ['/data/c_x/depedia_final_data/result_2.txt', '/data/c_x/depedia_final_data/result_3.txt', '/data/c_x/depedia_final_data/result_4.txt', '/data/c_x/depedia_final_data/result_5.txt', '/data/c_x/depedia_final_data/result_6.txt', '/data/c_x/depedia_final_data/result_7.txt']
# for p in path:
#     with open(p) as f:
#         for i in f.readlines():
#             try:
#                 h,r,t,q,a = i.strip().split('\t')
#             except:
#                 print(p, i)
#             has_result_triple.append((h,r,t))
# path = ['/data/c_x/depedia_final_data/2.txt', '/data/c_x/depedia_final_data/3.txt', '/data/c_x/depedia_final_data/4.txt', '/data/c_x/depedia_final_data/5.txt', '/data/c_x/depedia_final_data/6.txt', '/data/c_x/depedia_final_data/7.txt']

# all_triple = []
# for p in path:
#     with open(p) as f:
#         for i in f.readlines():
#             h, r, t= i.strip().split('\t')
#             all_triple.append((h,r,t))


# with open('/Users/machine/Downloads/1.txt') as f:
#     for i in f.readlines():
#         h, r, t = i.strip().split('\t')
#         all_triple.append((h,r,t))

# with open('/data/c_x/depedia_final_data/new_data/2_7_sy.txt', 'w+') as f:
#     for i in all_triple:
#         if i in has_result_triple:
#             continue
#         f.write(i[0]+'\t'+i[1]+'\t'+i[2]+'\n')

import math
all_triple = []
with open('/data/c_x/ChatGPT_Demo/kgc_dection/data/fb15k237/train_new.txt') as f:
    for i in f.readlines():
        h,r,t = i.strip().split('\t')
        all_triple.append((h,r,t))
max_num = math.ceil(len(all_triple) / 15000.0)

for i in range(max_num):
    start = i * 15000
    end = min(len(all_triple), (i + 1) * 15000)
    with open('/data/c_x/ChatGPT_Demo/kgc_dection/data/fb15k237/split/fb15k237_{}.txt'.format(i), 'w+') as f:
        for j in all_triple[start:end]:
            f.write(j[0]+'\t'+j[1]+'\t'+j[2]+'\n')

# import os
# all_triple = {}
# q_a = []
# for root, file, files in os.walk('/data/c_x/depedia_final_data/all_result'):
#     for i in files:
#         with open(os.path.join(root,i)) as f:
#             lines = f.readlines()
#             for ii in range(len(lines)):
#                 try:
#                     h,r,t,q,a = lines[ii].strip().split('\t')
#                     if (h,r,t) not in all_triple:
#                         all_triple[(h,r,t)] = [(q,a)]
#                     else:
#                         continue
#                     # q_a.append([q,a])
#                 except:
#                     print(i,ii)
#                     exit(-1)
# print(len((all_triple)))


# with open('/data/c_x/depedia_final_data/all_result/all_triple_chatgpt_result.txt', 'w+') as f:
#     for i in all_triple:
#         f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + all_triple[i][0][0] + '\t' + all_triple[i][0][1] + '\n')
# wrong_triple = []
# with open('/data/c_x/depedia_final_data/chatgpt_final_result/all_triple_chatgpt_result.txt', 'r') as f:
#     lines = f.readlines()
#     for i in lines:
#         h,r,t,q,a = i.strip().split('\t')
#         if a not in ['Yes.', 'No.', 'yes', 'no', 'yes.', 'no.']:
#             wrong_triple.append((h,r,t,q,'No.'))

# with open('/data/c_x/depedia_final_data/chatgpt_final_result/wrong.txt', 'w+') as f:
#     for i in wrong_triple:
#         f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\t' + i[4] + '\n')