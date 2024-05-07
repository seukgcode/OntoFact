import torch.nn as nn
import torch
import random
import numpy as np
class RL(nn.Module):
    def __init__(self, in_dim):
        super(RL, self).__init__()
        self.classifier = nn.Sequential()
        self.memory = ReplayBuffer_entity(int(10))
        classifier = []
        for _ in range(3 - 1):
            classifier.append(nn.Linear(in_dim, in_dim))
            classifier.append(nn.ReLU())
        classifier.append(nn.Linear(in_dim, 1))
        self.classifier= nn.Sequential(*classifier)
    
    def forward(self, samples):
        # samples: batch_size * dim
        score = torch.sigmoid(self.classifier(samples))
        # score = self.classifier(samples)
        return score


class ReplayBuffer_Ontology:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward = map(torch.stack, zip(*batch))
        return state, action.squeeze(1), next_state, reward

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer_entity:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, policy_ontology, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (policy_ontology, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        policy_ontology, reward = map(torch.stack, zip(*batch))
        return policy_ontology, reward

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, in_dim):
        super(Actor, self).__init__()
        self.classifier = nn.Sequential()
        classifier = []
        for _ in range(2 - 1):
            classifier.append(nn.Linear(in_dim, in_dim))
            classifier.append(nn.ReLU())
        classifier.append(nn.Linear(in_dim, in_dim))
        self.classifier= nn.Sequential(*classifier)
    def forward(self, samples):
        # samples: batch_size * dim
        action = self.classifier(samples)
        return action

class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.classifier = nn.Sequential()
        classifier = []
        hidden_dim = 2*in_dim
        for _ in range(3 - 1):
            classifier.append(nn.Linear(hidden_dim, hidden_dim))
            classifier.append(nn.ReLU())
        classifier.append(nn.Linear(hidden_dim, 1))
        self.classifier= nn.Sequential(*classifier)
    def forward(self, samples):
        x = self.classifier(samples)
        return x

class RL_ontology(nn.Module):
    def __init__(self, in_dim):
        super(RL_ontology, self).__init__()
        self.actor = Actor(in_dim)
        self.critic = Critic(in_dim)
        self.actor_target = Actor(in_dim)
        self.critic_target = Critic(in_dim)
        self.memory = ReplayBuffer_Ontology(int(10))
       
    
    def get_action(self, state):
        action = self.actor(state).unsqueeze(0).detach()
        return action
    # def forward(self, current_ontology_embedding):
    #     action = self.get_action(current_ontology_embedding)
    #     return  
        