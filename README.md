# Augmenting Few-shot Learning with Supervised Contrastive Learning


##  Contents
This repo contains the code for IEEE Access paper "Augmenting Few-shot Learning with Supervised Contrastive Learning". Our method augments the feature extractor using a contrastive learning technique. The main results in the paper can be reproduced with this repo.


## 1. Preparations
### 1.1 Preparing the dataset
Follow the instructions 1.1.1 from TIM https://github.com/mboudiaf/TIM to prepare Mini-ImageNet, CUB, and Tiered-ImageNet datasets. Modify the dataset path of the scripts below.

### 1.2 Preparing the model
We also provide pre-trained models. Please download the zip file at https://www.dropbox.com/s/ztp0s90jb8hyfxc/checkpoints.tar.gz?dl=0 and untar it.

### 1.3 Preparing the environment
Please see Dockerfile to prepare the environment.


## 2. Reproducing the results


### 2.1 Comparison to the state-of-the-art (Table 2, 3)


| 1 shot/5 shot |   Network   | mini-Imagenet |     CUB       | Tiered-Imagenet |
| 	   ---  |      ---    |      ---      |      ---      |    ---          |
| Ours-SPTA     |   MobileNet | 76.57 / 85.82 | 83.76 / 89.01 | 79.17 / 87.16   |
| Ours-SPTA     |   Resnet-18 | 78.83 / 87.76 | 88.81 / 93.11 | 81.16 / 88.43   |
| Ours-SPTA     |   WRN28-10  | 80.32 / 88.76 |               |                 |

We provide evaluate.py script to reproduce the results. For instance, execute:
```python
python evaluate.py --network resnet18 --dataset mini
```

### 2.2 Domain shift (Table 5)

| 1 shot/5 shot |   Network   | mini-Imagenet -> CUB  | tiered-Imagenet -> CUB |
| 	   ---  |      ---    |        ---            |	       ---             |
| Ours-SPTA     |   Resnet18  |    51.50 / 68.69      |    82.80 / 90.70       |

We provide domain\_shift.py script to reproduce the results. Please execute following commands.
```python
python domain_shift.py --dataset mini
python domain_shift.py --dataset tiered
```

### 2.3 Increasing the number of ways (Table 6)


| 1 shot/5 shot |  Network    |       10 ways     |       20 ways        |
| 	   ---  |     ---     |        ---        |	   ---           |
| Ours-SPTA     |   Resnet18  |   61.15 / 77.12   |     43.29 / 64.22    |

We provide mini\_10\_20\_ways.py script to reproduce the results. Please execute following commands.

```python
python mini_10_20_ways.py --way 10
python mini_10_20_ways.py --way 20
```

### 2.4 Runtime analysis (Figure 2)

We provide benchmark.py script to reproduce the results. Please execute following commands on Jetson TX2 platform.
```python
python benchmark.py --setting baseline
python benchmark.py --setting reduced
```


## 3. Train models (optional)

We provide pretrain.py script to train the network from scratch.
```python
python pretrain.py --dataset mini --network mobilenet 
python pretrain.py --dataset mini --network resnet18 --pretraining_batch_size 256
```


## Contact

For further questions or details, contact Taemin Lee (taemin.lee@snu.ac.kr)

## Acknowledgements

We would like to thank the authors from SimpleShot https://github.com/mileyan/simple_shot, LaplacianShot https://github.com/imtiazziko/LaplacianShot, and TIM https://github.com/mboudiaf/TIM for uploading their pre-trained models and their codes.
