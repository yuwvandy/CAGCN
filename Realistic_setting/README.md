# Realistic setting (Training/Validation/Testing splitted by time)
By default setting used in most literature, the data splitting is performed by randomly sampling for each node, certain amount of its neighbors to be training/validation/testing data. However, such splitting may cause the issue of data leakage where we use future links (training data contains future links) to predict past links (testing data contains past links). To handle this issue, we recollect the gowalla/yelp/amazon_book datasets and split them strictly according to time, which mimics the real scenarios.


**If you use this code, please consider citing:**
```linux
@inproceedings{CAGCN,
  author={Wang, Yu and Zhao, Yuying and Zhang, Yi and Derr, Tyler},
  title={Collaboration-Aware Graph Convolutional Networks for Recommendation Systems},
  booktitle={Proceedings of the ACM Web Conference 2023},
  year = {2023}
}
```

## Configuration
Similarly as the configuration used for experimental setting in the paper, the default version of python we use is 3.8.10. Please install all necessary python packages via:
```linux
- Pytorch 1.11.0 with Cuda 11.3
- Pytorch-geometric 2.0.4
- Torch-scatter 2.0.9
- Prettytable 3.2.0
```

## Data
We provide the preprocessing code in the notebook. Please follow instructions below for accessing the datasets and preprocessing them.
* **Gowalla**
  * Download gowalla.inter here and put it in ./data/gowalla/.
  * Run process_gowalla.ipynb and you should get train/val/test.txt
* **Yelp**
  * Download yelp_academic_dataset_review.json here and put it in ./data/yelp/.
  * Run process_yelp.ipynb and you should get train/val/test.txt
* **Amazon_book**
  * Download Books_5.json here and put it in ./data/amazon_book/.
  * Run process_amazon.ipynb and you should get train/val/test.txt



## Result
Here we list the performance of our models CAGCN(*) with different topological variants. To reproduce the performance and running time in the following Table, please run the following commands:
```linux
bash run_gowalla.sh
bash run_yelp.sh
bash run_amazon.sh
```
![](./tab_res_realistic.png)
