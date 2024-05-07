import numpy as np
import os
import pandas as pd
import itertools as iter
import torch
import logging

logger = logging.getLogger(__file__)

def get_root_logger(file_path, file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler(file_path + "/log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger

def build_datasets(args):
    basic_path = './dataset'
    dataset = args.dataset
    ontology_graph = []
    # entity_ontology = np.load(os.path.join(basic_path,dataset,'data','entity_ontology_mapping.npy'), allow_pickle = True).item()
    
    if args.unified:
        args.entity2id = np.load(os.path.join(basic_path,dataset,'embedding','instance_ontology_unified', 'instance_index_ents.npy'), allow_pickle=True).item()
        args.instance_rel2id = np.load(os.path.join(basic_path,dataset,'embedding','instance_ontology_unified', 'instance_index_rels.npy'), allow_pickle=True).item()
        args.ontology2id = np.load(os.path.join(basic_path,dataset,'embedding','instance_ontology_unified', 'ontology_index_ents.npy'),allow_pickle=True).item()
        args.preperty2id = np.load(os.path.join(basic_path,dataset,'embedding','instance_ontology_unified', 'ontology_index_rels.npy'), allow_pickle=True).item()
        args.id2ontology = {j:i for i,j in args.ontology2id.items()}
        args.id2preperty = {j:i for i,j in args.preperty2id.items()}
    else:
        ontology2id = pd.read_csv(os.path.join(basic_path,dataset,'data','entity_to_id.tsv'), sep='\t')
        ontology2id = dict(zip(ontology2id['label'], ontology2id['id']))
        preperty2id = pd.read_csv(os.path.join(basic_path,dataset,'data','relation_to_id.tsv'), sep='\t')
        preperty2id = dict(zip(preperty2id['label'], preperty2id['id']))
        args.ontology2id = ontology2id
        args.preperty2id = preperty2id
        args.id2ontology = {j:i for i,j in args.ontology2id.items()}
        args.id2preperty = {j:i for i,j in args.preperty2id.items()}
    with open(os.path.join(basic_path,dataset,'data','ontology.txt')) as f:
        for i in f.readlines():
            h_o, p, t_o = i.strip().split('\t')
            ontology_graph.append([args.ontology2id[h_o], args.preperty2id[p], args.ontology2id[t_o]])
    ontology_schema_mapping = np.load(os.path.join(basic_path,dataset,'data',args.datasetnpy), allow_pickle=True).item()
    # for i in ontology_schema_mapping:
    #     if len(ontology_schema_mapping[i])>100:
    #         index = np.random.choice(len(ontology_schema_mapping[i]), 100)
    #         ontology_schema_mapping[i] = np.array(ontology_schema_mapping[i])[index]
        # elif len(ontology_schema_mapping[i])<100:
        #     print(i)
    # with open(os.path.join(basic_path,dataset,'data','triple.txt')) as f:
    #     for i in f.readlines():
    #         h, r, t = i.strip().split('\t')
    #         h_o = entity_ontology[h]
    #         t_o = entity_ontology[t]
    #         for h_ , t_ in iter.product(h_o, t_o):
    #             if (h_, r, t_) not in ontology_schema_mapping:
    #                 ontology_schema_mapping[(h_, r, t_)] = [[h, r, t]]
    #             else:
    #                 ontology_schema_mapping[(h_, r, t_)].append([h, r, t])
    # np.save('ontology_schema_mapping.npy', ontology_schema_mapping)
    return ontology_schema_mapping, ontology_graph

def generate_embedding(args):
    if args.unified:
        entity_embedding = np.load(os.path.join('./dataset',args.dataset,'embedding','instance_ontology_unified', 'instance_entity.npy'))
        entity_rel_embedding = np.load(os.path.join('./dataset',args.dataset,'embedding','instance_ontology_unified', 'instance_r.npy'))
        ontology_embedding = np.load(os.path.join('./dataset',args.dataset,'embedding','instance_ontology_unified', 'ontology_entity.npy'))
        property_embedding = np.load(os.path.join('./dataset',args.dataset,'embedding','instance_ontology_unified', 'ontology_r.npy'))
        return entity_embedding, entity_rel_embedding, ontology_embedding, property_embedding
    else:
        ontology_embedding = np.load(os.path.join('./dataset',args.dataset,'embedding','entity_representations.npy'))
        property_embedding = np.load(os.path.join('./dataset',args.dataset,'embedding','relation_representations.npy'))
        return ontology_embedding, property_embedding



    