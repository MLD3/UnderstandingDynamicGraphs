#!/bin/sh

## declare gnn string arrays
declare -a hom_gnns=("sgc" "gcn" "gin" "gat")
declare -a het_gnns=("gsage" "gcnii" "fagcn")

## declare dataset string arrays
# declare -a datasets=("si_regular_20r_30t" "si_powerlaw_20r_30t" "si_random_20r_30t" "si_block_20r_30t" "si_uci_20r_29t" "si_eu_20r_41t" "si_alpha_20r_16t" "si_otc_20r_35t")
declare -a datasets=("si_math_20r_20t")

# launch hyperparameter search for each gnn and dataset 
for gnn in "${hom_gnns[@]}"
do
    for data in "${datasets[@]}"
    do
        python3 search.py --hidden 1 2 3 --model_name "$gnn" --data_name "$data"
    done
done

for gnn in "${het_gnns[@]}"
do
    for data in "${datasets[@]}"
    do
        python3 search.py --hidden 1 2 3 --model_name "$gnn" --data_name "$data"
    done
done
