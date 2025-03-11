# Understanding GNNs and Homophily in Dynamic Node Classification
[AISTATS 2025] Understanding GNNs and Homophily in Dynamic Node Classification

## Instructions for Reproducibility
1. In the `generation` directory, use the following command to download and preprocess datasets:
   ```bash
   source generate_graphs.sh
   ```
2. Once the datasets are downloaded and preprocessed, in the `root` directory, use the following command to train models and save results to the `evaluation` directory.
   ```bash
   source search.sh
   ```
3. After training the models and saving the results, load and visualize the results in the `analysis_social_sgnn.ipynb` or `analysis_synthetic_sgnn.ipynb` notebook located in the `evaluation` directory.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ItoKW25dynamic,
  author    = {Michael Ito and Danai Koutra and Jenna Wiens},
  title     = {Understanding GNNs and Homophily in Dynamic Node Classification},
  booktitle = {International Conference on Artificial Intelligence and Statistics},
  publisher = {PMLR},
  year      = {2025},
}
```
