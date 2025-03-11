import numpy as np
import subprocess
import argparse
import time
import os

def get_lrs(base, mult):
    lrs = []
    for b in base: 
        for m in mult: 
            lrs.append(b * m)
    return lrs

def format_command(command, hypers, gpu, model_idx, model_name, data_path, save_path): 
    args = [command]
    for key in hypers: 
        args.append(f' --{key} {hypers[key]}')
        
    args.append(f' --gpu {gpu}')
    args.append(f' --idx {model_idx}')
    args.append(f' --model_name {model_name}')
    args.append(f' --data_path {data_path}')
    args.append(f' --save_path {save_path}')
    
    string = ''
    for arg in args: 
        string += arg
        
    return string

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    
    # model hyperparameters
    parser.add_argument('--lrs_b', type=float, nargs='+', default=[.1, .01, .001, .0001])
    parser.add_argument('--lrs_m', type=int, nargs='+', default=[1, 5])
    parser.add_argument('--h_dim', type=int, nargs='+', default=[32])
    parser.add_argument('--num_layers', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--model_name', type=str)
    
    # dataset and search parameters
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--command', type=str, default='python3 main_train.py')

    # load arguments
    args = parser.parse_args()
    lrs_b = list(args.lrs_b)
    lrs_m = list(args.lrs_m)
    h_dim = list(args.h_dim)
    num_layers = list(args.num_layers)
    model_name = args.model_name
    gpus = list(args.gpus)
    data_name = args.data_name
    command = args.command
        
    # define available gpus, command, and parameters
    data_path = os.path.join('./data_proc/', data_name)
    save_path = os.path.join('./evaluation/', data_name)
    if not os.path.exists(save_path): 
        os.mkdir(save_path)

    # create search grid 
    hyperparameters = {'lr': get_lrs(lrs_b, lrs_m), 'h_dim': h_dim, 'num_layers': num_layers}
    grid = []
    for lr in hyperparameters['lr']: 
        for h in hyperparameters['h_dim']: 
            for nl in hyperparameters['num_layers']: 
                grid.append({'lr': lr, 'h_dim': h, 'num_layers': nl})

    # run search in parallel
    available_gpus = {gpu: subprocess.Popen('', shell=True) for gpu in gpus} # open a process for each available gpu
    model_idx = 0
    while len(grid) > 0: 
        for gpu in available_gpus.keys():
            if available_gpus[gpu].poll() == 0 and len(grid) > 0:
                hypers = grid.pop()
                command = format_command(command, hypers, gpu, model_idx, model_name, data_path, save_path)
                available_gpus[gpu] = subprocess.Popen(command, shell=True)
                model_idx += 1
                
                print(f'Training model {model_idx - 1}...')
                
        time.sleep(5) # wait 5 seconds to check for available gpus
    
    # wait for all processes to finish, so that next search doesn't overlap with current one
    for gpu in available_gpus.keys():
        available_gpus[gpu].wait()




