
def question_template(head,relation,tail):    
    if relation == '禁忌用药':
        question = "{}禁忌使用{}吗？".format(head,tail)
    if relation == '鉴别诊断':
        question = "在{}和{}之间，是否存在鉴别诊断？".format(head,tail)
    if relation == '有不良反应':
        question = "使用{}可能会产生{}这种不良反应吗？".format(head,tail)
    if relation == '有相互作用':
        question = "{}和{}之间有相互作用吗？".format(head,tail)
    if relation == '是一种':
        question = "{}属于{}吗？".format(head,tail)
    if relation == '有不良反应（反向）':
        question = "{}可能是由于使用{}产生的不良反应吗？".format(head,tail)
    if relation == '可导致（反向）':
        question = "{}可能是由于{}导致的吗？".format(head,tail)
    if relation == '可诊断（反向）':
        question = "{}可由{}诊断出来吗？".format(head,tail)
    if relation == '可治疗（反向）':
        question = "{}可由{}治疗吗？".format(head,tail)
    if relation == '可导致':
        question = "{}可能会导致{}吗？".format(head,tail)
    if relation == '可诊断':
        question = "通过{}可以诊断出{}吗？".format(head,tail)
    if relation == '可治疗':
        question = "{}可以用于治疗{}吗？".format(head,tail)
    return question

question = open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/chinese/20230714140951_chinese_only_triple_one_question.txt', 'w+')
with open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/chinese/20230714140951_chinese_only_triple_one.txt') as f:
    for i in f.readlines():
        h,r,t = i.strip().split('\t')
        question.write(h + '\t' + r + '\t' + t + '\t' + question_template(h,r,t) + '\n')

    