## Introduction
An inofficial PyTorch implementation of [SelfSAGCN: Self-Supervised Semantic Alignment for Graph Convolution Network, CVPR2021]

## Datasets
+ citeseer
+ cora
+ pubmed

| Dataset | Nodes | Edges | Features | Classes | #train | #val | #test |
| :---: | :---:| :---: | :---: | :---:| :---: | :---: | :---: |
| Citeseer | 3,327 | 4,732 | 3,703 | 6 | **120**(20 each class) | **500**(29, 86, 116, 106, 94, 69) | **1000**(77, 182, 181, 231, 169, 160) |
| Cora | 2,708 | 5,429 | 1,433 | 7 | **140**(20 each class) | **500**(61, 36, 78, 158, 81, 57, 29) | **1000**(130, 91, 144, 319, 149, 103, 64) |
| Pubmed | 19,717 | 44,338 | 500 | 3 | **60**(20 each class) | **500**(98, 194, 208) | **1000**(180, 413, 407) |


## Train
```
python demo.py

