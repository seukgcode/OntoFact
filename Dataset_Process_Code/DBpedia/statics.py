'''
|Author: RainbowCatSzy
|Date: 2023-04-04 12:50:49
|LastEditors: RainbowCatSzy
|LastEditTime: 2023-04-06 16:33:43
'''
def new_triple_generate():
    all_triple = []
    with open('/data/c_x/all_triples.txt', 'r') as f:
        for line in f.readlines():
            try:
                entity, relation, ontology = line.strip().split('\t')
            except:
                continue
            all_triple.append([entity, relation, ontology])
    print(len(all_triple))
    with open('/data/c_x/all_triples_new.txt', 'w+') as f:
        for t in all_triple:
            f.write(t[0] + '\t' + t[1]+ '\t' + t[2] + '\n')

def dbo_dbp_number():
    relation_set = set()
    with open('/data/c_x/ontology_dbo_dbp.txt', 'r') as f:
        for line in f.readlines():
            entity, relation, ontology = line.strip().split('\t')
            if "wikiPage" in relation:
                continue
            relation_set.add((entity,relation,ontology))
    
    with open('/data/c_x/ontology_dbo_dbp_new_no_wikiPage.txt', 'w+') as f:
        for t in relation_set:
            f.write(t[0] + '\t' + t[1]+ '\t' + t[2] + '\n')
    
    

def new_triple_no_wikipage_generate():
    all_triple = []
    with open('/data/c_x/all_triples.txt', 'r') as f:
        for line in f.readlines():
            try:
                entity, relation, ontology = line.strip().split('\t')
            except:
                continue
            if 'wikiPage' in relation:
                continue
            all_triple.append([entity, relation, ontology])
    print(len(all_triple))
    with open('/data/c_x/all_triples_new_no_wikipage.txt', 'w+') as f:
        for t in all_triple:
            f.write(t[0] + '\t' + t[1]+ '\t' + t[2] + '\n')

def relation_statics_triple():
    all_relation = set()
    with open('/data/c_x/all_triples_new_no_wikipage_25157878.txt', 'r') as f:
        for line in f.readlines():
            try:
                entity, relation, ontology = line.strip().split('\t')
            except:
                continue
            if 'wikiPage' in relation:
                continue
            all_relation.add(relation)
    print(len(all_relation))
def new_triple_no_wikipage_generate_no_singleOrtwo_character():
    all_triple = []
    with open('/data/c_x/all_triples.txt', 'r') as f:
        for line in f.readlines():
            try:
                entity, relation, ontology = line.strip().split('\t')
            except:
                continue
            if 'wikiPage' in relation or len(relation)<=2 or 'logo' in relation or 'image' in relation:
                continue
            all_triple.append([entity, relation, ontology])
    print(len(all_triple))
    with open('/data/c_x/all_triples_new_no_wikipage_no_SingleTwoCharacter.txt', 'w+') as f:
        for t in all_triple:
            f.write(t[0] + '\t' + t[1]+ '\t' + t[2] + '\n')

def new_triple_Relation_sameWith_Cake():
    pre_relation = []
    with open('/data/c_x/realtion2.txt', 'r') as f:
        pre_relation = [line.strip() for line in f.readlines()]
    all_triple = set()
    with open('/data/c_x/all_triples.txt', 'r') as f:
        for line in f.readlines():
            try:
                entity, relation, ontology = line.strip().split('\t')
            except:
                continue
            if relation in pre_relation:
                all_triple.add((entity, relation, ontology))
    print(len(all_triple))
    with open('/data/c_x/all_triples_Relation_num600_{}.txt'.format(len(all_triple)), 'w+') as f:
        for t in all_triple:
            f.write(t[0] + '\t' + t[1]+ '\t' + t[2] + '\n')

def dbo_dbp_relation_number():
    relation_set = set()
    with open('ontology_dbo_dbp_new_no_wikiPage_SameWith_PreFixRelation.txt', 'r') as f:
        for line in f.readlines():
            entity, relation, ontology = line.strip().split('\t')
            relation_set.add(ontology)
    
    with open('ontology_dbo_dbp_new_no_wikiPage_SameWith_PreFixRelation_ontology.txt', 'w+') as f:
        for t in relation_set:
            f.write(t + '\n')

def dbo_dbp_number_SameWith_PreFixRelation():
    dbo_dbp = set()
    relation = []
    with open('/data/c_x/relation_600.txt', 'r') as f:
        for t in f.readlines():
            relation.append(t.strip())
    with open('/data/c_x/ontology_dbo_dbp_new_no_wikiPage.txt', 'r') as f:
        for line in f.readlines():
            entity, r, ontology = line.strip().split('\t')
            if r in relation:
                dbo_dbp.add((entity,r,ontology))
    print(len(dbo_dbp))
    with open('ontology_dbo_dbp_new_no_wikiPage_SameWith_PreFixRelation.txt', 'w+') as f:
        f.writelines([i[0] + '\t' + i[1] + '\t' + i[2] + '\n' for i in dbo_dbp])
    

dbo_dbp_relation_number()