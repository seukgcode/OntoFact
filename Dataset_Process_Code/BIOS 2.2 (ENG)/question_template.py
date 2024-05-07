
def question_template(head,relation,tail):    
    if relation == 'ddx':
        question = "Is there a differential diagnosis between {} and {}?".format(head,tail)
    if relation == 'contraindication':
        question = "Is {} contraindicated for {}?".format(head,tail)
    if relation == 'has adverse effect':
        question = "Is it possible that the use of {} may have an adverse effect such as {}?".format(head,tail)
    if relation == 'interacts with':
        question = "Is there an interaction between {} and {}?".format(head,tail)
    if relation == 'is a':
        question = "Can {} be classified as {}?".format(head,tail)
    if relation == 'is adverse effect of':
        question = "May {} be an adverse effect of using {}?".format(head,tail)
    if relation == 'may be caused by':
        question = "May {} be caused by {}?".format(head,tail)
    if relation == 'may be diagnosed by':
        question = "May {} be diagnosed by {}?".format(head,tail)
    if relation == 'may be treated by':
        question = "May {} be treated by {}?".format(head,tail)
    if relation == 'may cause':
        question = "May {} cause {}?".format(head,tail)
    if relation == 'may diagnose':
        question = "May {} diagnose {}?".format(head,tail)
    if relation == 'may treat': # 1
        question = "May {} treat {}?".format(head,tail)
    return question

def question_template_NA(head,relation,tail):    
    if relation == 'ddx':
        question = "Is there a differential diagnosis between {} and {}?".format(head,'N/A')
    if relation == 'contraindication':
        question = "Is {} contraindicated for {}?".format(head,'N/A')
    if relation == 'has adverse effect':
        question = "Is it possible that the use of {} may have an adverse effect such as {}?".format(head,'N/A')
    if relation == 'interacts with':
        question = "Is there an interaction between {} and {}?".format(head,'N/A')
    if relation == 'is a':
        question = "Can {} be classified as {}?".format(head,'N/A')
    if relation == 'is adverse effect of':
        question = "May {} be an adverse effect of using {}?".format(head,'N/A')
    if relation == 'may be caused by':
        question = "May {} be caused by {}?".format(head,'N/A')
    if relation == 'may be diagnosed by':
        question = "May {} be diagnosed by {}?".format(head,'N/A')
    if relation == 'may be treated by':
        question = "May {} be treated by {}?".format(head,'N/A')
    if relation == 'may cause':
        question = "May {} cause {}?".format(head,'N/A')
    if relation == 'may diagnose':
        question = "May {} diagnose {}?".format(head,'N/A')
    if relation == 'may treat': # 1
        question = "May {} treat {}?".format(head,'N/A')
    return question

question = open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/english/20230714140951_english_only_triple_one_question_NA.txt', 'w+')
with open('/data01/c_x/ChatGPT_Demo/domain-specific/mecial/bios_v2.2_release/findal_data/english/20230714140951_english_only_triple_one.txt') as f:
    for i in f.readlines():
        h,r,t = i.strip().split('\t')
        question.write(h + '\t' + r + '\t' + t + '\t' + question_template_NA(h,r,t) + '\n')

    