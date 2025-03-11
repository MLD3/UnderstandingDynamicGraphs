# python modules
import os
import pickle as pkl
import networkx as nx
import numpy as np

from scipy import stats

def compute_transition_homophily(g): 
    '''
    Computes graph total homophily and transition homophily for g
        - As in Ito et al. 2023, first compute the transition homohpily for positives and negatives, then return the mean of both as the total homophily
    '''
    hom_pos, hom_neg = [], []
    for node in g.nodes: # iterate through all nodes in g
        if g.nodes[node]['positive'] == 0: # only measure homophily for nodes we are predicting on, so positive = 0
            if g.nodes[node]['exposure'] == 0: # y_t+1 is 0, so count number of matching 0s
                if g.degree[node] == 0: # account for isolated nodes
                    node_degree = 1
                else: 
                    node_degree = g.degree[node]

                node_homophily = 1 # count the self loop
                for neighbor in g.neighbors(node): # iterate over node's neighbors and count matching future label 0s
                    if g.nodes[neighbor]['positive'] == 0: 
                        node_homophily += 1
                hom_neg.append(node_homophily / (node_degree+1))
                
            else: # y_t+1 is 1, so count number of matching 1s
                if g.degree[node] == 0: # account for isolated nodes
                    node_degree = 1
                else: 
                    node_degree = g.degree[node]

                node_homophily = 0 # don't count the self-loop since future label is 1
                for neighbor in g.neighbors(node): # iterate over node's neighbors and count matching future label 1s
                    if g.nodes[neighbor]['positive'] == 1: 
                        node_homophily += 1
                hom_pos.append(node_homophily / (node_degree+1))

    # if there are nodes in both classes, set total homophily as the mean between the two, else return np.inf
    if len(hom_pos) > 0 and len(hom_neg) > 0: 
        t_hom = (np.mean(hom_pos) + np.mean(hom_neg)) / 2
    else: 
        t_hom = np.inf
        
    return t_hom

def compute_label_homophily(g):
    '''
    Computes graph label homophily for over all nodes in the snapshot of g
        - As in Ito et al. 2023, we use only information in the snapshot, so we rely on current infection status 
    '''
    homs = []
    for node in g.nodes: # iterate through all nodes in g
        if g.degree[node] == 0: # account for isolated nodes
            node_degree = 1
        else: 
            node_degree = g.degree[node]

        node_homophily = 0
        node_positive = g.nodes[node]['positive']
        for neighbor in g.neighbors(node): # count number of matching current infecteds
            if g.nodes[neighbor]['positive'] == node_positive: 
                node_homophily += 1
                
        homs.append(node_homophily / node_degree)
    hom_mean = np.mean(homs)
        
    return hom_mean

def compute_perfs_stat_mean(path, model, trials, n_repeats, timesteps, top_k): 
    '''
    Gets the best performing model according to val_auc stored in the performance dictionary across all static graph model trials
    '''
    aucs, preds, val_aucs, test_aucs, test_errs, hypers = [[] for i in range(trials)], [[] for i in range(trials)], [], [], [], []
    for i in range(trials): 
        id_path = os.path.join(path, f'{model}_results_{i}.pkl')
        if os.path.exists(id_path): 
            with open(id_path, 'rb') as file: 
                perf, args = pkl.load(file)
                aucs[i].append(perf['aucs'])
                val_aucs.append(perf['val_auc'])
                test_aucs.append(perf['test_auc'])
                test_errs.append(perf['test_err'])
                hypers.append((args.lr, args.hidden))
        else: 
            val_aucs.append(-1)

    perf_matrix = []
    for k in range(top_k): 
        best_id = np.argsort(val_aucs)[-k]
        perfs = aucs[best_id][0]
        perf = []
        for i in range(n_repeats): 
            perf.append(list(perfs[(i * timesteps): (i + 1) * timesteps]))    
            
        perf = np.array(perf)
        perf_matrix.append(perf[:, :, None])

    perf_matrix = np.mean(np.concatenate(perf_matrix, axis=-1), axis=-1)
    perf_m, perf_e = compute_means_errs(perf_matrix)
    
    print(f'{model}: {test_aucs[best_id]:.2f} $\pm$ {test_errs[best_id]:.2f}')
    
    return perf_m, perf_e, perf_matrix

def compute_perfs_stat(path, model, trials, n_repeats, timesteps, top_k): 
    '''
    Gets the best performing model according to val_auc stored in the performance dictionary across all static graph model trials
    '''
    aucs, preds, val_aucs, test_aucs, test_errs, hypers = [[] for i in range(trials)], [[] for i in range(trials)], [], [], [], []
    for i in range(trials): 
        id_path = os.path.join(path, f'{model}_results_{i}.pkl')
        if os.path.exists(id_path): 
            with open(id_path, 'rb') as file: 
                perf, args = pkl.load(file)
                aucs[i].append(perf['aucs'])
                val_aucs.append(perf['val_auc'])
                test_aucs.append(perf['test_auc'])
                test_errs.append(perf['test_err'])
                hypers.append((args.lr, args.hidden))
        else: 
            val_aucs.append(-1)

    best_id = np.argsort(val_aucs)[-top_k]
    perfs = aucs[best_id][0]
    perf = []
    for i in range(n_repeats): 
        perf.append(list(perfs[(i * timesteps): (i + 1) * timesteps]))

    perf = np.array(perf)
    perf_m, perf_e = compute_means_errs(perf)
    
    print(f'{model}: {test_aucs[best_id]:.2f} $\pm$ {test_errs[best_id]:.2f}')
    
    return perf_m, perf_e, perf

def compute_perfs_dyn(path, model, trials, top_k): 
    '''
    Gets the best performing model according to val_auc stored in the performance dictionary across all dynamic graph model trials
    '''
    aucs_matrices, val_aucs, test_aucs, test_errs, hypers = [], [], [], [], []
    for i in range(trials): 
        id_path = os.path.join(path, f'{model}_results_{i}.pkl')
        if os.path.exists(id_path): 
            with open(id_path, 'rb') as file: 
                perf, args = pkl.load(file)
                aucs_matrices.append(perf['auc_matrix'])
                val_aucs.append(perf['val_auc'])
                test_aucs.append(perf['test_auc'])
                test_errs.append(perf['test_err'])
                hypers.append((args.lr, args.h_dim))
        else: 
            val_aucs.append(-1)

    best_id = np.argsort(val_aucs)[-top_k]
    best_auc_matrix = aucs_matrices[best_id]
    best_auc_m, best_auc_e = compute_means_errs(best_auc_matrix)
    
    print(f'{model}: {test_aucs[best_id]:.2f} $\pm$ {test_errs[best_id]:.2f}')
    
    return best_auc_m, best_auc_e, best_auc_matrix

def compute_means_errs(x): 
    '''
    Computes the mean and 95% confidence interval across axis 1 for a matrix X (N_test x Timesteps)
    '''
    x_mean = np.zeros((x.shape[1], ))
    x_errs = np.zeros((x.shape[1], ))
    for i in range(x.shape[1]): 
        x_real = x[np.isfinite(x[:, i]) == 1, i]
        x_mean[i] = np.mean(x_real)
        x_errs[i] = 1.96 * np.std(x_real) / np.sqrt(x_real.shape[0])
    
    return x_mean, x_errs

def compute_correlation(x, y, n_infecteds, threshold, axis=0): 
    '''
    Computes the correlation between x and y, masking out timesteps where n_infecteds < threshold
    '''
    x = np.array(x)
    rhos = np.zeros((x.shape[0], ))

    if axis == 0: 
        for i in range(x.shape[0]):
            mask = (np.isfinite(x[i, :]).astype(int) + np.isfinite(y[i, :]).astype(int) + (n_infecteds[i, :] >= threshold).astype(int)) > 2
            rhos[i] = stats.spearmanr(x[i, :][mask], y[i, :][mask])[0]
    else: 
        for i in range(x.shape[0]):
            mask = (np.isfinite(x[:, i]).astype(int) + np.isfinite(y[:, i]).astype(int) + (n_infecteds[:, i] >= threshold).astype(int)) > 2
            rhos[i] = stats.spearmanr(x[:, i][mask], y[:, i][mask])[0]
            
    return np.mean(rhos[np.isfinite(rhos)]), np.std(rhos[np.isfinite(rhos)])

def normalize(x): 
    '''
    Min-max normalizes x in the interval 0, 1
    '''
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def sis_susceptible_rate(g): 
    '''
    Measure the rate of susceptible nodes in g for either SI or SIS model
    '''
    rate = 0
    for node in g.nodes: 
        if g.nodes[node]['positive'] == 0 and g.nodes[node]['exposure'] == 0: 
            rate += 1
    return rate/g.number_of_nodes()

def sis_infected_rate(g): 
    '''
    Measure the rate of infected nodes in g for either SI or SIS model
    '''
    rate = 0
    for node in g.nodes: 
        if g.nodes[node]['exposure'] == 1: 
            rate += 1
    return rate/g.number_of_nodes()




