# README.md

```wiki
@author HELLORPG
@date 2022.7.5
```

本项目考虑构建一个基于 PyTorch 实现的视觉学习框架，目前主要包括了如下功能：
- [x] 模型训练
- [x] 模型评估
- [x] 日志输出
- [x] 进度条展示
- [x] 分布式模式

下载 [MNIST](https://github.com/HELLORPG/CV-Framework/releases/download/v1.0.0/MNIST.zip) 数据集之后无需解压，默认放在`./dataset/MNIST/`路径下，运行`run.sh`中的所有语句就可以进行功能测试。


## Project Tree
```bash
.
├── configs   # 配置文件以及相关函数
│   ├── __init__.py
│   ├── resnet18_mnist.yaml
│   └── utils.py
├── data      # 数据操作，包括构造 Dataset DataLoader
│   ├── __init__.py
│   ├── mnist.py
│   └── utils.py
├── log       # 日志操作，包括了日志输出、存储和计算
│   ├── __init__.py
│   ├── logger.py
│   └── log.py
├── models    # 网络结构
│   ├── __init__.py
│   ├── resnet18.py
│   └── utils.py
├── utils     # utils
│   ├── __init__.py
│   └── utils.py
├── LICENSE
├── README.md
├── engine.py
├── main.py
└── run.sh
```


## 配置文件

整个项目的配置信息（例如模型具体层数、训练方式、各类超参数等）都放在文件夹 [configs](./configs) 中。
其中配置信息以`.yaml`文件的形式存储，样例如下：
```yaml
MODE: train
DEVICE: cuda
GPUS: [0, 1, 2, 3, 4, 5, 6, 7]

DATA:
  DATASET: MNIST
  DATA_PATH: ./dataset/MNIST/
  CLASS_NUM: 10
  NUM_WORKERS: 0

MODEL:
  PRETRAINED: False

TRAIN:
  LR: 0.0001
  BATCH_SIZE: 32
  EPOCHS: 10
  RESUME:
    RESUME_MODEL:
    START_EPOCH:
    CHECKPOINT_OPTIM_STATE: True
  SCHEDULER:
    MILESTONES: [6, 8]
    GAMMA: 0.5

EVAL:
  EVAL_MODEL:

OUTPUTS:
  OUTPUTS_DIR: ./outputs/

DISTRIBUTED:
  USE_DISTRIBUTED: False
```

