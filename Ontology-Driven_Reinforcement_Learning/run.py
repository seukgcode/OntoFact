import argparse
from util import build_datasets, generate_embedding, get_root_logger
import logging
import time
import os
from model import ChatGPT_Detection
from RL_DQN import DQN 
# from traverseEnv import TraverseEnv
from RL import RL
import warnings
import torch
import numpy as np
import random
warnings.filterwarnings("ignore", category=Warning)

# def set_seed(seed=2023):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     random.seed(seed)  # Python random module.

#     # torch.set_deterministic(True)
#     # torch.backends.cudnn.enabled = False 
#     # torch.backends.cudnn.benchmark = False
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",  type=str, default='yago_4.5')
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument("-c", "--cluster", help="the type of cluster algorithm", type=str, default='K-means')
    parser.add_argument("-u", "--unified", help="the type of cluster algorithm", action='store_false')
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--LLMType", type=str, default='Bloom')
    parser.add_argument("--qt", help='Question Type such as: triple、template、manual_template', type=str, default='triple')
    parser.add_argument("--train_type", help='Question Type such as: triple、template、manual_template', type=str, default='has_answer')
    parser.add_argument("--datasetnpy", help='Question Type such as: triple、template、manual_template', type=str, default='triple')

    
    args = parser.parse_args()

    return args

def train(args, ontology_schema_mapping, ontology_graph, ontology_embedding, property_embedding, entity_embedding, entity_rel_embedding):
    # RL_ontology_model = RL(300).to(args.device)
    RL_entity_model = RL(900).to(args.device) # dbpedia 900 yago 700 cn-dbpedia 900
    model = ChatGPT_Detection(args,ontology_schema_mapping, ontology_graph, ontology_embedding, property_embedding,RL_entity=RL_entity_model, entity_embedding= entity_embedding, entity_rel_embedding= entity_rel_embedding)
    # model()
    
    # model.LLM_detection_no_cluster(RL)
    model.LLM_detection_path()
    

if __name__ == "__main__":
    # set_seed()
    working_dir = './experiment/{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(working_dir)
    logger = get_root_logger(working_dir)
    logger.warning('ChatGPT')
    args = parse_args()
    args.device = 'cuda:{}'.format(args.device)
    args.working_dir = working_dir
    logger.warning('build_datasets')
    ontology_schema_mapping, ontology_graph = build_datasets(args=args)
    if args.unified:
        entity_embedding, entity_rel_embedding, ontology_embedding, property_embedding = generate_embedding(args=args)
    else:    
        ontology_embedding, property_embedding = generate_embedding(args=args)
    
    logger.warning('Train')
    train(args, ontology_schema_mapping, ontology_graph, ontology_embedding, property_embedding, entity_embedding, entity_rel_embedding)
    