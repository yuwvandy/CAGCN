# CAGCN
This repository is an official PyTorch(Geometric) implementation of CAGCN in ["Collaboration-Aware Graph Convolutional Networks for
Recommendation Systems"]().

**If you use this code, please consider citing:**
```linux
@inproceedings{fairview,
  author={Wang, Yu and Zhao, Yuying and Zhang, Yi and Derr, Tyler},
  title={Collaboration-Aware Graph Convolutional Networks for Recommendation Systems},
  booktitle={Arxiv},
  year = {2022}
}
```

## Motivation
Most GNN-based recommendation systems perform message-passing by directly applying traditional GNN-convolutions. How does the message-passing captures collaborative effect? Are the message-passing by traditional GNN-convolutions really beneficial to users' ranking? This paper will take you to demystify the collaborations captured by traditional GNN-convolutions.

Specifically, we demystify the collaborations captured by LightGCN by answering the following two questions:
- **How does the message-passing captures collaborative effect?** We find that the L-layer LightGCN-based message-passing captures and leverages collaborations between nodes within L-hops neighborhoods of the center user and item. We also theoretically derive the strength of the captured collaborations.
![](./img/collaboration.png)

- **Does the captured collaboration really help the prediction of users' ranking?** We propose a new recommendation-tailored topological metric, Common Interacted Ratio (CIR), and empirically find that higher CIR leads to more benefits to users' ranking.
![](./img/CIR.png)

Based on our theoretical and empirical analysis, we incorporate CIR into the message-passing and ultimately propose a novel class of Collaboration-Aware Graph Convolutional Networks, namely Collaboration-Aware Graph Convolutional Network (CAGCN) and its augmented version (CAGCN*), both of which are able to **selectively pass information of neighbors based on their CIR via the designed Collaboration-Aware Graph Convolution**. The whole framework is as follows:
![](./img/framework.png)


We also propose a brandly new type of graph isomorphism for bipartite graphs and theoretically prove that **the designed CAGCN\* can go beyond 1-WL test in distinguishing subtree-isomorphic(subgraph-isomorphic) graphs yet not bipartite-subgraph-isomorphic graphs**.

![](./img/isomorphism.png)


## Configuration
The default version of python we use is 3.8.10. Please install all necessary python packages via:
```linux
- Pytorch 1.11.0 with Cuda 11.3
- Pytorch-geometric 2.0.4
- Torch-scatter 2.0.9
- Prettytable 3.2.0
```

## Result
Here we list the performance of our models CAGCN(*) with different topological variants. To reproduce the table, please run the following command:
```linux
bash run_gowalla.sh
bash run_yelp.sh
bash run_amazon.sh
bash run_ml1m.sh
bash run_loseit.sh
bash run_worldnews.sh
```
![](./img/restable.png)

## Acknowledgement: The code is developed based on part of the code in the following papers:
```linux
[1] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.
[2] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua. Neural graph collaborative filtering. SIGIR 2019.
[3] Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu Wang, Jie Tang. MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. KDD 2021.
```
