import sys
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

import utils

class SIStatic(): 
    def __init__(self, g, timesteps=20, inf_alpha=1, inf_beta=1, sus_alpha=1, sus_beta=1): 
        '''
        inf_alpha: alpha parameter for infectious feature
        inf_beta: beta parameter for infectious feature
        '''
        # graph parameters
        self.timesteps = timesteps
        
        # infection parameters
        self.inf_alpha, self.inf_beta = inf_alpha, inf_beta
        self.sus_alpha, self.sus_beta = sus_alpha, sus_beta
        self.N_INIT_NODES = 1
        
        self.g_init = self.initialize_graph(g)
        self.graphs = self.simulate_infection(self.g_init)
        
    def initialize_graph(self, g):
        infected_nodes = np.random.choice(g.nodes, size=(self.N_INIT_NODES, ))
        for node in g.nodes: 
            infectious = np.random.beta(self.inf_alpha, self.inf_beta)
            susceptible = np.random.beta(self.sus_alpha, self.sus_beta)
            if node in infected_nodes:
                attrs = {'positive': 1, 'exposure': 1, 'susceptible': susceptible, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]
            else:
                attrs = {'positive': 0, 'exposure': 0, 'susceptible': susceptible, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]
                    
        return g

    def simulate_infection(self, g):
        graphs = []
        for i in range(self.timesteps): 
            g = self.simulate_infection_single(g)
            g = self.generate_features(g)
            graphs.append(g)

        return graphs

    def simulate_infection_single(self, g):
        next_g = g.copy()
        
        # update next_g attributes: set nodes with exposure = 1 to positive = 1
        for node in next_g.nodes: 
            if next_g.nodes[node]['exposure'] == 1: 
                next_g.nodes[node]['positive'] = 1
        
        for node in next_g.nodes:
            if g.nodes[node]['exposure'] == 0:
                infected = 0
                p_uninfected = 1
                for neighbor in g.neighbors(node): 
                    p_neighbor = g.nodes[node]['susceptible'] * g.nodes[neighbor]['infectious'] * g.nodes[neighbor]['exposure']
                    p_uninfected *= 1 - p_neighbor
                    infected += np.random.binomial(1, p_neighbor)
                    
                if infected > 0: 
                    next_g.nodes[node]['exposure'] = 1
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
                else: 
                    next_g.nodes[node]['exposure'] = 0
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
            else: 
                next_g.nodes[node]['p_infected'] = -1 # for debugging purposes

        return next_g

    def generate_features(self, g): 
        for node in g.nodes: 
            features = []
            if g.nodes[node]['positive'] == 1: 
                features.append(1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])
            else: 
                features.append(-1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])

            g.nodes[node]['features'] = np.array(features)

        return g
    
class SISStatic(): 
    def __init__(self, g, timesteps=20, patience=10, inf_alpha=1, inf_beta=1, sus_alpha=1, sus_beta=1, rec_alpha=1, rec_beta=1): 
        '''
        inf_alpha: alpha parameter for infectious feature
        inf_beta: beta parameter for infectious feature
        '''
        # graph parameters
        self.timesteps = timesteps
        self.patience = patience
        
        # infection parameters
        self.inf_alpha, self.inf_beta = inf_alpha, inf_beta
        self.sus_alpha, self.sus_beta = sus_alpha, sus_beta
        self.rec_alpha, self.rec_beta = rec_alpha, rec_beta
        self.N_INIT_NODES = 10
        
        self.g_init = self.initialize_graph(g)
        self.graphs = self.simulate_infection(self.g_init)
        
    def initialize_graph(self, g):
        infected_nodes = np.random.choice(g.nodes, size=(self.N_INIT_NODES, ))
        for node in g.nodes: 
            infectious = np.random.beta(self.inf_alpha, self.inf_beta)
            susceptible = np.random.beta(self.sus_alpha, self.sus_beta)
            recovery = np.random.beta(self.rec_alpha, self.rec_beta)
            if node in infected_nodes:
                attrs = {'positive': 1, 'exposure': 1, 'susceptible': susceptible, 'recovery': recovery, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]
            else:
                attrs = {'positive': 0, 'exposure': 0, 'susceptible': susceptible, 'recovery': recovery, 'infectious': infectious}
                for key in attrs: 
                    g.nodes[node][key] = attrs[key]
                    
        return g

    def simulate_infection(self, g): 
        for i in range(self.patience): 
            graphs = []
            for i in range(self.timesteps): 
                g = self.simulate_infection_single(g)
                g = self.generate_features(g)
                graphs.append(g)

            if utils.sis_infected_rate(graphs[-1]) > utils.sis_susceptible_rate(graphs[-1]): 
                break
            elif i < self.patience - 1: 
                g = self.initialize_graph(g)
                
        return graphs

    def simulate_infection_single(self, g):
        next_g = g.copy()
        
        # update next_g attributes: set nodes with exposure = 1 to positive = 1
        for node in next_g.nodes: 
            if next_g.nodes[node]['exposure'] == 1: 
                next_g.nodes[node]['positive'] = 1
        
        for node in next_g.nodes:
            if g.nodes[node]['exposure'] == 0:
                infected = 0
                p_uninfected = 1
                for neighbor in g.neighbors(node): 
                    p_neighbor = g.nodes[node]['susceptible'] * g.nodes[neighbor]['infectious'] * g.nodes[neighbor]['exposure']
                    p_uninfected *= 1 - p_neighbor
                    infected += np.random.binomial(1, p_neighbor)
                    
                if infected: 
                    next_g.nodes[node]['exposure'] = 1
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
                else: 
                    next_g.nodes[node]['exposure'] = 0
                    next_g.nodes[node]['p_infected'] = 1 - p_uninfected
            else: 
                recovered = np.random.binomial(1, g.nodes[node]['recovery'])
                if recovered: 
                    next_g.nodes[node]['exposure'] = 0
                    next_g.nodes[node]['positive'] = 0
                    next_g.nodes[node]['p_infected'] = -1
                else: 
                    next_g.nodes[node]['p_infected'] = -1

        return next_g

    def generate_features(self, g): 
        for node in g.nodes: 
            features = []
            if g.nodes[node]['positive'] == 1: 
                features.append(1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])
                features.append(g.nodes[node]['recovery'])
            else: 
                features.append(-1)
                features.append(g.nodes[node]['infectious'])
                features.append(g.nodes[node]['susceptible'])
                features.append(g.nodes[node]['recovery'])

            g.nodes[node]['features'] = np.array(features)

        return g
