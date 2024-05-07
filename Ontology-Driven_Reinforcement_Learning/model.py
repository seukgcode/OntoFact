import torch
from torch import nn
import numpy as np
from scipy.spatial.distance import euclidean
from time import sleep
import asyncio
from RL import RL_ontology
import random
import faiss
import sklearn
import time
sep = '-'*30

class ChatGPT_Detection(nn.Module):
    def __init__(self, args, ontology_schema_mapping, ontology_graph, ontology_embedding, property_embedding, RL_entity, entity_embedding, entity_rel_embedding):
        super(ChatGPT_Detection, self).__init__()
        self.args=args
        self.cluset_algorithm = args.cluster
        self.ontology_schema_mapping = ontology_schema_mapping
        self.ontology_graph = np.array(ontology_graph)
        self.ontology_embedding = ontology_embedding
        self.property_embedding = property_embedding
        # self.T5 = T5Model(args)
        if args.train_type != 'has_answer':
            self.LLM = getattr(__import__('LLM'), self.args.LLMType)(args)
        # self.T0 = T0Model(args)
        self.RL_ontology = RL_ontology(300).to(self.args.device)  # 英文300
        self.RL_entity = RL_entity
        # self.ontology_optimizer = torch.optim.Adam(self.RL_ontology.parameters(), lr=0.001)
        self.ontology_optimizer_actor = torch.optim.Adam(self.RL_ontology.actor.parameters(), lr=0.0002) # 0.001
        self.ontology_optimizer_critic = torch.optim.Adam(self.RL_ontology.critic.parameters(), lr=0.0002) # 0.001
        self.entity_optimizer = torch.optim.Adam(self.RL_entity.parameters(),lr=2e-4) # 1e-4
        self.entity_embedding = entity_embedding
        self.entity_rel_embedding = entity_rel_embedding

        self.entity_criterion = torch.nn.BCELoss()
    

    def calculate_cluster_center_index(self, cluster_centers_, labels_, n_clusters, ontology_triple_embedding):
        index = []
        for i in range(n_clusters):
            cluster_sample = torch.where(torch.tensor(labels_) == i)[0]
            min_index = torch.argmin(torch.tensor([euclidean(ontology_triple_embedding[idx], torch.tensor(cluster_centers_[i])) for idx in cluster_sample]))
            index.append(cluster_sample[min_index])
        return index

    def calculate_start_ontology(self, index):
        # 0 = Yes. 1 = No. 2 = Uncertain.
        yes_start_file = open(self.args.working_dir + '/start.txt', 'w+')
        no_start_file = open(self.args.working_dir + '/no_start.txt', 'w+')
        uncertain_start_file = open(self.args.working_dir + '/uncertain_start.txt', 'w+')
        other_start_file = open(self.args.working_dir + '/other_start.txt', 'w+')
        result = torch.zeros(len(index),3)
        for ii in range(len(index)):
            i = index[ii]
            ontology_triple_question = self.ontology_graph[i]
            triple = self.ontology_schema_mapping[(self.args.id2ontology[ontology_triple_question[0]], self.args.id2preperty[ontology_triple_question[1]], self.args.id2ontology[ontology_triple_question[2]])]
            for t in triple:
                response = self.chatgpt.request(t)
                if response == 'Yes.' or response == 'yes' or response == 'Yes' or response == 'yes.':
                    yes_start_file.write(t[0]+'\t'+t[1]+'\t'+t[2]+'\n')
                    result[ii][0] += 1
                elif response == 'No.' or response == 'no' or response == 'No' or response == 'no.':
                    no_start_file.write(t[0]+'\t'+t[1]+'\t'+t[2]+'\n')
                    result[ii][1] += 1
                elif response == 'Uncertain.' or response == 'uncertain' or response == 'Uncertain' or response == 'uncertain.':
                    uncertain_start_file.write(t[0]+'\t'+t[1]+'\t'+t[2]+'\n')
                    result[ii][2] += 1
                else:
                    other_start_file.write(t[0]+'\t'+t[1]+'\t'+t[2]+'\t'+response+'\n')
        yes_start_file.close()
        no_start_file.close()
        uncertain_start_file.close()
        return result
    
    def batch_calculate_start_ontology(self, index):
        # 0 = Yes. 1 = No. 2 = Uncertain.
        all_result_file = open(self.args.working_dir + '/async_result.txt', 'w+')
        result = torch.zeros(len(index),3)
        for ii in range(len(index)):
            i = index[ii]
            ontology_triple_question = self.ontology_graph[i]
            triple = self.ontology_schema_mapping[(self.args.id2ontology[ontology_triple_question[0]], self.args.id2preperty[ontology_triple_question[1]], self.args.id2ontology[ontology_triple_question[2]])]
            n_batch = len(triple) // 3 + (len(triple) % 3 > 0)
            for i in range(n_batch):
                start = i*3
                end = min(len(triple), (i+1)*3)
                batch_data = np.array(triple)[np.arange(start, end),:]
                batch_response = asyncio.run(self.chatgpt.request_async(batch_data))
                print(batch_response)
                all_result_file.writelines([i[0][0]+'\t'+i[0][1]+'\t'+i[0][2]+'\t'+i[1]+'\n' for i in batch_response])   
        all_result_file.close()
        return result
                
    def forward(self):
        ontology_h_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,0]])
        ontology_r_embedding = torch.tensor(self.property_embedding[self.ontology_graph[:,1]])
        ontology_t_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,2]])
        ontology_triple_embedding = torch.cat([ontology_h_embedding, ontology_r_embedding, ontology_t_embedding], dim=-1)
        # cluster_centers_, labels_, n_clusters = self.cluster_model(ontology_triple_embedding)
        cluster_centers_ = np.load('/data/c_x/ChatGPT_Demo/cluster_centers.npy')
        labels_ =  np.load('/data/c_x/ChatGPT_Demo/cluster_label.npy')
        n_clusters = 100
        # center node
        index = self.calculate_cluster_center_index(cluster_centers_, labels_, n_clusters, ontology_triple_embedding)
        # result = self.calculate_start_ontology(index)
        result = self.batch_calculate_start_ontology(index)
        print(result)
    
    def LLM_calculate_start_ontology(self, index):
        all_result_file = open(self.args.working_dir + '/LLM_result.txt', 'w+')
        result = torch.zeros(len(index),3)
        # 0 = Yes. 1 = No. 2 = Uncertain.
        for ii in range(len(index)):
            i = index[ii]
            ontology_triple_question = self.ontology_graph[i]
            triple = self.ontology_schema_mapping[(self.args.id2ontology[ontology_triple_question[0]], self.args.id2preperty[ontology_triple_question[1]], self.args.id2ontology[ontology_triple_question[2]])]
            n_batch = len(triple) // 100 + (len(triple) % 100 > 0)
            for i in range(n_batch):
                start = i*100
                end = min(len(triple), (i+1)*100)
                batch_data = np.array(triple)[np.arange(start, end),:]
                batch_response = self.T5.generation(batch_data)
                # batch_response = asyncio.run(self.chatgpt.request_async(batch_data))
                result[ii,0] += len(np.where(np.array(batch_response)[:,1]=='Yes')[0])
                result[ii,1] += len(np.where(np.array(batch_response)[:,1]=='No')[0])
                result[ii,2] += len(np.where(np.array(batch_response)[:,1]=='Uncertain')[0])
                all_result_file.writelines([i[0][0]+'\t'+i[0][1]+'\t'+i[0][2]+'\t'+i[1]+'\n' for i in batch_response])   
        all_result_file.close()
        return result
    
    def LLM_detection(self, RL):
        label = torch.zeros(self.ontology_graph.shape[0])
        ontology_h_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,0]])
        ontology_r_embedding = torch.tensor(self.property_embedding[self.ontology_graph[:,1]])
        ontology_t_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,2]])
        ontology_triple_embedding = torch.cat([ontology_h_embedding, ontology_r_embedding, ontology_t_embedding], dim=-1)
        # cluster_centers_, labels_, n_clusters = self.cluster_model(ontology_triple_embedding)
        cluster_centers_ = np.load('/data/c_x/ChatGPT_Demo/cluster_centers.npy')
        labels_ =  np.load('/data/c_x/ChatGPT_Demo/cluster_label.npy')
        n_clusters = 100
        # center node
        index = self.calculate_cluster_center_index(cluster_centers_, labels_, n_clusters, ontology_triple_embedding)
        label[index] = 1
        # result = self.calculate_start_ontology(index)
        # result = self.batch_calculate_start_ontology(index)
        result = self.LLM_calculate_start_ontology(index)
        error_probablity = result[:,1] / result.sum(dim=-1)

    def LLM_detection_yes_no(self, triple, all_result_file, ontology_embedding):

        
        result = torch.zeros(1,3)
        # 0 = Yes. 1 = No. 2 = Uncertain.
        bath_size = 20
        n_batch = len(triple) // bath_size + (len(triple) % bath_size > 0)
        consume_entity_reward = torch.zeros(len(triple)).to(self.args.device)
        entity_reward = []
        policy_n = [] 
        all_action = []
        batch_label_all = []
        # all_entity_probablity = []
        for i in range(n_batch):
            start = i*bath_size
            end = min(len(triple), (i+1)*bath_size)
            batch_data = np.array(triple)[np.arange(start, end),:]
            batch_data_index = np.array([[self.args.entity2id[i[0]],self.args.instance_rel2id[i[1]],self.args.entity2id[i[2]]] for i in batch_data])
            batch_data_embedding = torch.cat([torch.tensor(self.entity_embedding[batch_data_index[:,0]]), torch.tensor(self.entity_rel_embedding[batch_data_index[:,1]]), torch.tensor(self.entity_embedding[batch_data_index[:,2]])], dim=-1).to(self.args.device)
            batch_data_embedding = torch.cat([ontology_embedding.expand(len(batch_data),-1), batch_data_embedding], dim=-1)
            entity_probablity = self.RL_entity(batch_data_embedding)
            # all_entity_probablity.append(entity_probablity)
            batch_response = np.array(self.LLM.generation(batch_data))
            batch_label = np.ones_like(np.array(batch_response)[:,1],dtype=np.float32)
            batch_response[:,1] = np.array([x.lower() if isinstance(x, str) else x for x in batch_response[:,1]])
            batch_label[np.array(batch_response)[:,1] == 'yes'] = -1.0
            batch_label[np.array(batch_response)[:,1] == 'no'] = 2.0
            entity_probablity_action = torch.ones_like(entity_probablity)
            action = torch.ones_like(entity_probablity)
            action[entity_probablity>=0.5] = 1
            action[entity_probablity<0.5] = 0
            all_action.append(action)
            # 每个时刻的 p_sita1
            policy_n.append(action * entity_probablity + (1-action) * (1-entity_probablity))
            entity_probablity_action[entity_probablity>=0.5] = 1
            entity_probablity_action[entity_probablity<0.5] = -1
            # 每个时刻的奖励
            batch_label = torch.tensor(batch_label).to(self.args.device)
            batch_label_all.append(batch_label)
            entity_reward_batch = entity_probablity_action * batch_label.unsqueeze(-1)
            entity_reward.append(entity_reward_batch)
            # entity_score = (entity_probablity_action * torch.tensor(batch_label).to(self.args.device).unsqueeze(-1) * torch.log(policy_n)).sum()
            # batch_response = asyncio.run(self.chatgpt.request_async(batch_data))
            result[0,0] += len(np.where(np.array(batch_response)[:,1]=='yes')[0])
            result[0,1] += len(np.where(np.array(batch_response)[:,1]=='no')[0])
            result[0,2] += len(np.where(np.array(batch_response)[:,1]=='uncertain')[0])
            all_result_file.writelines([i[0][0]+'\t'+i[0][1]+'\t'+i[0][2]+'\t'+i[1]+'\n' for i in batch_response])
        entity_reward = torch.cat(entity_reward,dim=0)
        policy_n = torch.cat(policy_n,dim=0)
        all_action = torch.cat(all_action,dim=0).squeeze(-1)
        batch_label_all = torch.cat(batch_label_all,dim=0)
        batch_label_all[batch_label_all==-1] = 0
        # all_entity_probablity = torch.cat(all_entity_probablity, dim=0)
        R = 0
        for i in range(0,len(entity_reward)):
            R = entity_reward[i][0] + 0.95 * R
            consume_entity_reward[i] = R
        # if len(consume_entity_reward)>1:
        #     consume_entity_reward = (consume_entity_reward - consume_entity_reward.mean()) / (consume_entity_reward.std() + 1e-8)
        # print('actula_error:',(all_action).sum() / len(all_action))
        return result, consume_entity_reward, policy_n, (all_action).sum() / len(all_action)
    
    def LLM_detection_yes_no_answer(self, triple, all_result_file, ontology_embedding):

        
        result = torch.zeros(1,3)
        # 0 = Yes. 1 = No. 2 = Uncertain.
        bath_size = 100
        n_batch = len(triple) // bath_size + (len(triple) % bath_size > 0)
        consume_entity_reward = torch.zeros(len(triple)).to(self.args.device)
        entity_reward = []
        policy_n = [] 
        all_action = []
        batch_label_all = []
        all_entity_probablity = []
        for i in range(n_batch):
            start = i*bath_size
            end = min(len(triple), (i+1)*bath_size)
            batch_data = np.array(triple)[np.arange(start, end),:]
            batch_data_index = np.array([[self.args.entity2id[i[0]],self.args.instance_rel2id[i[1]],self.args.entity2id[i[2]]] for i in batch_data])
            print(self.entity_rel_embedding.shape,self.entity_embedding.shape)
            batch_data_embedding = torch.cat([torch.tensor(self.entity_embedding[batch_data_index[:,0]]), torch.tensor(self.entity_rel_embedding[batch_data_index[:,1]]), torch.tensor(self.entity_embedding[batch_data_index[:,2]])], dim=-1).to(self.args.device)
            # batch_data_embedding = torch.cat([ontology_embedding.expand(len(batch_data),-1), batch_data_embedding], dim=-1)
            # print(batch_data_embedding.shape)
            entity_probablity = self.RL_entity(batch_data_embedding)
            all_entity_probablity.append(entity_probablity)
            # batch_response = np.array(self.LLM.generation(batch_data))
            batch_response = np.array(batch_data)[:,3]
            batch_label = np.ones_like(np.array(batch_response),dtype=np.float32)
            batch_label[np.array(batch_response) == 'yes'] = -1.0
            batch_label[np.array(batch_response) == 'no'] = 1.0
            entity_probablity_action = torch.ones_like(entity_probablity)
            action = torch.ones_like(entity_probablity)
            action[entity_probablity>=0.5] = 1
            action[entity_probablity<0.5] = 0
            all_action.append(action)
            # 每个时刻的 p_sita1
            policy_n.append(action * entity_probablity + (1-action) * (1-entity_probablity))
            entity_probablity_action[entity_probablity>=0.5] = 1
            entity_probablity_action[entity_probablity<0.5] = -1
            # 每个时刻的奖励
            batch_label = torch.tensor(batch_label).to(self.args.device)
            batch_label_all.append(batch_label)
            entity_reward_batch = entity_probablity_action * batch_label.unsqueeze(-1)
            entity_reward.append(entity_reward_batch)
            # entity_score = (entity_probablity_action * torch.tensor(batch_label).to(self.args.device).unsqueeze(-1) * torch.log(policy_n)).sum()
            # batch_response = asyncio.run(self.chatgpt.request_async(batch_data))
            result[0,0] += len(np.where(np.array(batch_response)=='yes')[0])
            result[0,1] += len(np.where(np.array(batch_response)=='no')[0])
            result[0,2] += len(np.where(np.array(batch_response)=='uncertain')[0])
            # all_result_file.writelines([i[0][0]+'\t'+i[0][1]+'\t'+i[0][2]+'\t'+i[1]+'\n' for i in batch_response])
        entity_reward = torch.cat(entity_reward,dim=0)
        policy_n = torch.cat(policy_n,dim=0)
        all_action = torch.cat(all_action,dim=0).squeeze(-1)
        batch_label_all = torch.cat(batch_label_all,dim=0)
        batch_label_all[batch_label_all==-1] = 0
        all_entity_probablity = torch.cat(all_entity_probablity, dim=0)
        R = 0
        for i in range(0,len(entity_reward)):
            R = entity_reward[i][0] + 0.90 * R
            consume_entity_reward[i] = R
        # if len(consume_entity_reward)>1:
        #     consume_entity_reward = (consume_entity_reward - consume_entity_reward.mean()) / (consume_entity_reward.std() + 1e-8)
        # print('actula_error:',(all_action).sum() / len(all_action))
        return result, consume_entity_reward, policy_n, (all_action).sum() / len(all_action), all_entity_probablity, batch_label_all, all_action


    

    def LLM_detection_yes_no_backup(self, triple, all_result_file, ontology_embedding):
    
        
        result = torch.zeros(1,3)
        # 0 = Yes. 1 = No. 2 = Uncertain.
        batch_size = 20
        
        n_batch = len(triple) // batch_size + (len(triple) % batch_size > 0)
        consume_entity_reward = torch.zeros(len(triple)).to(self.args.device)
        entity_reward = []
        policy_n = [] 
        all_action = []
        all_entity_probablity = []
        for i in range(n_batch):
            start = i*batch_size
            end = min(len(triple), (i+1)*batch_size)
            batch_data = np.array(triple)[np.arange(start, end),:]
            batch_data_index = np.array([[self.args.entity2id[i[0]],self.args.instance_rel2id[i[1]],self.args.entity2id[i[2]]] for i in batch_data])
            batch_data_embedding = torch.cat([torch.tensor(self.entity_embedding[batch_data_index[:,0]]), torch.tensor(self.entity_rel_embedding[batch_data_index[:,1]]), torch.tensor(self.entity_embedding[batch_data_index[:,2]])], dim=-1).to(self.args.device)
            batch_data_embedding = torch.cat([ontology_embedding.unsqueeze(0).expand(len(batch_data),-1), batch_data_embedding], dim=-1)
            entity_probablity = self.RL_entity(batch_data_embedding)
            # entity_yes_no_predict = 
            batch_response = np.array(self.LLM.generation(batch_data))
            batch_label = np.ones_like(np.array(batch_response)[:,1],dtype=np.float32)
            batch_response[:,1] = np.array([x.lower() if isinstance(x, str) else x for x in batch_response[:,1]])
            batch_label[np.array(batch_response)[:,1] == 'yes'] = -1.0
            batch_label[np.array(batch_response)[:,1] == 'no'] = 1.0
            entity_probablity_action = torch.ones_like(entity_probablity)
            action = torch.ones_like(entity_probablity)
            action[entity_probablity>=0.5] = 1
            action[entity_probablity<0.5] = 0
            all_action.append(action)
            # 每个时刻的 p_sita1
            policy_n.append(action * entity_probablity + (1-action) * (1-entity_probablity))
            entity_probablity_action[entity_probablity>=0.5] = 1
            entity_probablity_action[entity_probablity<0.5] = -1
            # 每个时刻的奖励
            entity_reward_batch = entity_probablity_action * torch.tensor(batch_label).to(self.args.device).unsqueeze(-1)
            entity_reward.append(entity_reward_batch)
            # entity_score = (entity_probablity_action * torch.tensor(batch_label).to(self.args.device).unsqueeze(-1) * torch.log(policy_n)).sum()
            # batch_response = asyncio.run(self.chatgpt.request_async(batch_data))
            result[0,0] += len(np.where(np.array(batch_response)[:,1]=='yes')[0])
            result[0,1] += len(np.where(np.array(batch_response)[:,1]=='no')[0])
            result[0,2] += len(np.where(np.array(batch_response)[:,1]=='uncertain')[0])
            all_result_file.writelines([i[0][0]+'\t'+i[0][1]+'\t'+i[0][2]+'\t'+i[1]+'\n' for i in batch_response])
        entity_reward = torch.cat(entity_reward,dim=0)
        policy_n = torch.cat(policy_n,dim=0)
        all_action = torch.cat(all_action,dim=0).squeeze(-1)
        R = 0
        for i in range(len(entity_reward)-1,-1,-1):
            R = entity_reward[i][0] + 0.95 * R
            consume_entity_reward[i] = R
        if len(consume_entity_reward)>1:
            consume_entity_reward = (consume_entity_reward - consume_entity_reward.mean()) / (consume_entity_reward.std() + 1e-8)
        return result, consume_entity_reward, policy_n, all_action.sum() / len(all_action)

    

    def LLM_detection_no_cluster(self):
        all_result_file = open(self.args.working_dir + '/LLM_result.txt', 'a+')
        ontology_result_file = open(self.args.working_dir + '/LLM_ontology_result.txt', 'a+')
        label = torch.zeros(self.ontology_graph.shape[0])
        ontology_h_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,0]])
        ontology_r_embedding = torch.tensor(self.property_embedding[self.ontology_graph[:,1]])
        ontology_t_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,2]])
        ontology_triple_embedding = torch.cat([ontology_h_embedding, ontology_r_embedding, ontology_t_embedding], dim=-1)
        alpha = torch.tensor(1.0).to(self.args.device)
        # # cluster_centers_, labels_, n_clusters = self.cluster_model(ontology_triple_embedding)
        # cluster_centers_ = np.load('/data/c_x/ChatGPT_Demo/cluster_centers.npy')
        # labels_ =  np.load('/data/c_x/ChatGPT_Demo/cluster_label.npy')
        # n_clusters = 100
        # # center node
        # index = self.calculate_cluster_center_index(cluster_centers_, labels_, n_clusters, ontology_triple_embedding)
        # label[index] = 1
        # # result = self.calculate_start_ontology(index)
        # # result = self.batch_calculate_start_ontology(index)
        # result = self.LLM_calculate_start_ontology(index)

        for i in range(self.ontology_graph.shape[0]):

            ontology_triple_question = self.ontology_graph[i]
            triple = self.ontology_schema_mapping[(self.args.id2ontology[ontology_triple_question[0]], self.args.id2preperty[ontology_triple_question[1]], self.args.id2ontology[ontology_triple_question[2]])]
            if len(triple) == 0:
                continue

            # self.optimizer.zero_grad()
            
            label[i] = 1

            ontology_embedding = ontology_triple_embedding[i].to(self.args.device)
            probablity = self.RL_ontology(ontology_embedding)

            action = 1 if probablity>=0.5 else 0
            policy_ontology = action * probablity + (1 - action) * (1 - probablity)



            yes_no_result, consume_entity_reward, policy_entity, entity_error_probablity = self.LLM_detection_yes_no(i, triple, all_result_file=all_result_file, ontology_embedding= ontology_embedding)

            # 错误率是一个非负的，我们更喜欢看到超过50%事实性错误的情况，直接减个 0.5
            error_probablity = (yes_no_result[:,1] / yes_no_result.sum(dim=-1)).to(self.args.device)
            print(yes_no_result)



            action_score = (torch.log(policy_ontology * policy_entity.squeeze(-1)) * (consume_entity_reward + alpha * (error_probablity - 0.5))).sum()

            loss = -1 * action_score
            loss.backward()
            self.optimizer.step()
            print(loss.item(), error_probablity.item()>0.5, probablity.item()>0.5, entity_error_probablity.item())
            ontology_result_file.writelines([self.args.id2ontology[self.ontology_graph[i][0]] + '\t' + self.args.id2preperty[self.ontology_graph[i][1]] + '\t' +self.args.id2ontology[ self.ontology_graph[i][2]] + '\t' + str(error_probablity.detach().cpu()) + '\n'])


        ontology_result_file.close()
        all_result_file.close()
    
    def find_next_state(self, search, action):
        D, I = search.search(action, 1)
        return I[0][0]
    
    def update(self, ontology_batch_size, entity_batch_size):
        # 更新 ontology
        if len(self.RL_ontology.memory) >= ontology_batch_size:
            # self.ontology_optimizer.zero_grad()
            state_embedding, action, next_state_embedding, ontology_reward = self.RL_ontology.memory.sample(ontology_batch_size)
            # target_q = ontology_reward
            target_q = ontology_reward + 0.95 * self.RL_ontology.critic_target(torch.cat([next_state_embedding, self.RL_ontology.actor_target(next_state_embedding)],dim=-1))
            q = self.RL_ontology.critic(torch.cat([state_embedding,action.to(self.args.device)],dim=-1))

            critic_loss = torch.nn.functional.mse_loss(q, target_q)
            self.ontology_optimizer_critic.zero_grad()
            critic_loss.backward()
            self.ontology_optimizer_critic.step()

            actor_loss = -self.RL_ontology.critic(torch.cat([state_embedding, self.RL_ontology.actor(state_embedding)],dim=-1)).mean()
            self.ontology_optimizer_actor.zero_grad()
            # ontology_loss = critic_loss + actor_loss
            actor_loss.backward()
            self.ontology_optimizer_actor.step()
            print(sep)
            print('critic_loss:', critic_loss)
            print('actor_loss:', actor_loss)
            print('ontology_reward:', ontology_reward)
            print('ontology_reward:', q)
            print(sep)
            for target_param, param in zip(self.RL_ontology.actor_target.parameters(), self.RL_ontology.actor.parameters()):
                target_param.data.copy_(0.001 * param.data + (1 - 0.001) * target_param.data)
            for target_param, param in zip(self.RL_ontology.critic_target.parameters(), self.RL_ontology.critic.parameters()):
                target_param.data.copy_(0.001 * param.data + (1 - 0.001) * target_param.data)
        # # 更新 entity
        # if len(self.memory) >= entity_batch_size:
        #     policy_entity, entity_reward = self.RL_entity.memory.sample(entity_batch_size)
        #     entity_score = -1 * (torch.log(policy_entity.squeeze(-1)) * entity_reward).sum()
        #     entity_score.backward()
        #     self.entity_optimizer.step()
        


        
        

    def LLM_detection_path(self):
        all_result_file = open(self.args.working_dir + '/LLM_result.txt', 'a+')
        ontology_result_file = open(self.args.working_dir + '/LLM_ontology_result.txt', 'a+')
        ontology_h_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,0]])
        ontology_r_embedding = torch.tensor(self.property_embedding[self.ontology_graph[:,1]])
        ontology_t_embedding = torch.tensor(self.ontology_embedding[self.ontology_graph[:,2]])
        ontology_triple_embedding = torch.cat([ontology_h_embedding, ontology_r_embedding, ontology_t_embedding], dim=-1)
        alpha = torch.tensor(12.0).to(self.args.device)
        
        start_index = random.sample(range(self.ontology_graph.shape[0]), 1)
        ontology_index = np.arange(self.ontology_graph.shape[0])
        visited_ontology = np.array(start_index)
        all_reward = 0
        
        state = start_index[0]

        self.RL_ontology.train()
        self.RL_entity.train()
        cnt = 0
        num = 0
        torch.save(self.RL_ontology.state_dict(),self.args.working_dir + '/RL_ontology.pt')
        torch.save(self.RL_entity.state_dict(),self.args.working_dir + '/RL_entity.pt')
        all_count = 0
        alpha_p = 0.9
        size_ontology = 1/3 * self.ontology_graph.shape[0]
        for i in range(size_ontology):
            num+=1
            ontology_new_index = np.delete(ontology_index,visited_ontology,0)
            # selected_ontology_embedding = ontology_triple_embedding.detach().cpu().numpy()[ontology_new_index]
            selected_ontology_embedding = ontology_triple_embedding.detach().cpu().numpy()[ontology_new_index]
            # print(selected_ontology_embedding.shape)
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatL2(300)  # 英文 300  中文700
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index_flat.add(selected_ontology_embedding)

            ontology_triple_question = self.ontology_graph[state]
            triple = self.ontology_schema_mapping[(self.args.id2ontology[ontology_triple_question[0]], self.args.id2preperty[ontology_triple_question[1]], self.args.id2ontology[ontology_triple_question[2]])]
            if len(triple) == 0:
                continue

            ontology_embedding = ontology_triple_embedding[state].to(self.args.device)

            alpha_choice = np.random.choice(np.array([0,1]),size=1,p=np.array([1-alpha_p,alpha_p]))
            # alpha_choice = 0
            if not alpha_choice:
                action = self.RL_ontology.get_action(ontology_embedding)
                next_state = self.find_next_state(gpu_index_flat, action.cpu().numpy())
            else:
                # 附近的点
                next_state = self.find_next_state(gpu_index_flat, ontology_embedding.unsqueeze(0).cpu().numpy())
                action = ontology_embedding.unsqueeze(0)
            next_state = ontology_new_index[next_state]
            next_state_embedding = ontology_triple_embedding[next_state].to(self.args.device)
            visited_ontology = np.append(visited_ontology, next_state)
            # LLM_detection_yes_no_answer 不需要实时生成答案
            yes_no_result, consume_entity_reward, policy_entity, entity_error_probablity, all_p, batch_all, all_action = self.LLM_detection_yes_no_answer(triple, all_result_file=all_result_file, ontology_embedding= ontology_embedding)
            error_probablity = (yes_no_result[:,1] / yes_no_result.sum(dim=-1)).to(self.args.device)

            # 本体的强化学习奖励
            ontology_reward = error_probablity - 0.5

            # 真标签
            if error_probablity >= 0.5:
                entity_reward = consume_entity_reward + alpha * (entity_error_probablity - 0.5)
                # ontology_reward = error_probablity - 0.5
                # entity_reward = consume_entity_reward
            else:
                entity_reward = consume_entity_reward + alpha * (0.5 - entity_error_probablity)
                # entity_reward = consume_entity_reward
            # auc = sklearn.metrics.roc_auc_score(batch_all.detach().cpu(), all_p.squeeze(-1).detach().cpu())
            # print('auc:', auc)
            # print(batch_all.shape, all_action.shape)
            # f1_score = sklearn.metrics.f1_score(batch_all.detach().cpu(), all_action.detach().cpu(), average = 'binary')
            # print('f1:', f1_score)
            # entity_reward = consume_entity_reward + alpha * (f1_score - 0.5)
            # entity_reward = consume_entity_reward
        
            distance = torch.nn.functional.pairwise_distance(next_state_embedding, ontology_embedding, p=2)
            self.RL_ontology.memory.push(ontology_embedding, action, next_state_embedding, ontology_reward)
            # self.RL_entity.memory.push(policy_entity, ontology_reward)

            self.update(2,200)
            self.entity_optimizer.zero_grad()
            # entity_loss = self.entity_criterion(all_p.squeeze(-1), batch_all)
            # entity_loss = self.multilabel_categorical_crossentropy(all_p.reshape(1,-1),batch_all.unsqueeze(-1).reshape(1,-1))
            entity_loss = -1 * (torch.log(policy_entity.squeeze(-1)) * entity_reward).sum() 
            
            entity_loss.backward()
            self.entity_optimizer.step()
            all_reward+=entity_reward.sum()
            print('entity_loss:',entity_loss)
            # print('distance', distance)
            
            
            if (error_probablity >= 0.5 and entity_error_probablity >= 0.5) or (error_probablity < 0.5 and entity_error_probablity < 0.5):
                cnt+=1
            
            # if error_probablity.detach().cpu().item()>=0.5 and entity_error_probablity >= 0.5:
            #     cnt+=1
            print(cnt * 1.0 / num)
            print(all_reward, error_probablity.item()>0.5, entity_error_probablity.item())
            ontology_result_file.writelines([self.args.id2ontology[self.ontology_graph[state][0]] + '\t' + self.args.id2preperty[self.ontology_graph[state][1]] + '\t' +self.args.id2ontology[self.ontology_graph[state][2]] + '\t' + str(error_probablity.detach().cpu().item()) + '\t' + str(entity_error_probablity.detach().cpu().item()) + '\t' + str(distance.detach().cpu().item()) +  '\n'])
            state = next_state
            alpha_p = max(0.1, alpha_p-(0.8/1000)*i)
        torch.save(self.RL_ontology.state_dict(),self.args.working_dir + '/RL_ontology.pt')
        torch.save(self.RL_entity.state_dict(),self.args.working_dir + '/RL_entity.pt')
        self.RL_ontology.eval()
        self.RL_entity.eval()
        for i in range(self.ontology_graph.shape[0]-size_ontology + 1):
            start_time = time.time()
            ontology_new_index = np.delete(ontology_index,visited_ontology,0)
            selected_ontology_embedding = ontology_triple_embedding.detach().cpu().numpy()[ontology_new_index]
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatL2(300)
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index_flat.add(selected_ontology_embedding)

            ontology_triple_question = self.ontology_graph[state]
            triple = self.ontology_schema_mapping[(self.args.id2ontology[ontology_triple_question[0]], self.args.id2preperty[ontology_triple_question[1]], self.args.id2ontology[ontology_triple_question[2]])]
            if len(triple) == 0:
                continue

            ontology_embedding = ontology_triple_embedding[state].to(self.args.device)

            action = self.RL_ontology.get_action(ontology_embedding)
            next_state = self.find_next_state(gpu_index_flat, action.cpu().numpy())
            if next_state == -1:
                next_state = -1
            else:
                next_state = ontology_new_index[next_state]
                next_state_embedding = ontology_triple_embedding[next_state].to(self.args.device)
            visited_ontology = np.append(visited_ontology, next_state)
            # LLM_detection_yes_no_answer 不需要实时生成答案
            yes_no_result, consume_entity_reward, policy_entity, entity_error_probablity, all_p, batch_all, all_action = self.LLM_detection_yes_no_answer(triple, all_result_file=all_result_file, ontology_embedding= ontology_embedding)

            error_probablity = (yes_no_result[:,1] / yes_no_result.sum(dim=-1)).to(self.args.device)

            ontology_reward = error_probablity - 0.5

            # 真标签
            if error_probablity > 0.5:
                entity_reward = consume_entity_reward + alpha * (entity_error_probablity - 0.5)
                # ontology_reward = error_probablity - 0.5
                # entity_reward = consume_entity_reward
            else:
                entity_reward = consume_entity_reward + alpha * (0.5 - entity_error_probablity)
                # entity_reward = consume_entity_reward
            
            # entity_reward = consume_entity_reward

            

            # self.RL_ontology.memory.push(ontology_embedding, action, next_state_embedding, ontology_reward)
            
            print(error_probablity.item()>0.5, entity_error_probablity.item())
            ontology_result_file.writelines([self.args.id2ontology[self.ontology_graph[state][0]] + '\t' + self.args.id2preperty[self.ontology_graph[state][1]] + '\t' +self.args.id2ontology[self.ontology_graph[state][2]] + '\t' + str(error_probablity.detach().cpu().item()) + '\t' + str(entity_error_probablity.detach().cpu().item()) + '\n'])
            state = next_state
            print(time.time()-start_time)
            if next_state == -1:
                break

        ontology_result_file.close()
        all_result_file.close()
        
        
        

        