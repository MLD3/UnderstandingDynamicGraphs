#!/bin/sh

## synthetic graphs
python3 main_generate.py --graph_name regular --param 3 --inf_params 6 1 --sus_params 2 2 --gpu 0 
python3 main_generate.py --graph_name powerlaw --param 3 --inf_params 3 6 --sus_params 3 6 --gpu 0
python3 main_generate.py --graph_name random --param .01 --inf_params 2 5 --sus_params 2 5 --gpu 0
python3 main_generate.py --graph_name block --param .01 --inf_params 3 6 --sus_params 3 6 --gpu 4 

## real-world graphs
python3 main_generate.py --graph_name uci --real 0 --window_size 2000 --inf_params 2 4 --sus_params 2 2 --gpu 0
python3 main_generate.py --graph_name eu --real 1 --window_size 5000 --inf_params 2 6 --sus_params 2 6 --gpu 0
python3 main_generate.py --graph_name math --real 2 --window_size 12000 --inf_params 2 2 --sus_params 2 2 --gpu 0
python3 main_generate.py --graph_name alpha --real 3 --window_size 100 --inf_params 4 2 --sus_params 4 2 --gpu 0
python3 main_generate.py --graph_name otc --real 4 --window_size 1000 --inf_params 4 2 --sus_params 4 2 --gpu 0
