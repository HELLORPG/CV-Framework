# README.md

本项目考虑构建一个基于 PyTorch 实现的视觉学习框架，目前主要包括了如下功能：
- [x] 模型训练
- [x] 模型评估
- [x] 日志输出
- [x] 进度条展示
- [x] 分布式模式

下载 [MNIST](https://github.com/HELLORPG/CV-Framework/releases/download/v1.0.0/MNIST.zip) 数据集之后无需解压，默认放在`./dataset/MNIST/`路径下，运行`run.sh`中的所有语句就可以进行功能测试。


## Primary Components

### Main script
In file [main.py](./main.py), it's responsible for starting this project running.
Use `python` and other parameters to make it process different job, as follows:
```bash
# For model training:
python main.py --mode train --outputs-dir ./outputs/ --use-distributed False --exp-name train
# For model evaluation:
python main.py --mode eval --outputs-dir ./outputs/ --use-distributed False --eval-model ./outputs/train/checkpoint_4.pth --exp-name eval
```

### Model files
In package [models](./models), you can use the method 'build_model()' in [models/\_\_init\_\_.py](./models/__init__.py) to get the whole network model ([ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) in this project as an example).

Usually, different modules of the overall model will be written in different files. Since the sample network is tiny, it only has a single module in [models/resnet18.py](./models/resnet18.py). In [models/utils.py](./models/utils.py), there are also some utils code about model, such as `save_checkpoint()` and `load_checkpoint()`. For some tasks and models, their loss function (criterion) may be complex (for example, [MOTR](https://github.com/megvii-research/MOTR)), thus they require a separate code file to implement.

### Data processing
Data file reading, data transformation and data loader (Dataset, Sampler, DataLoader) are all included in package [data](./data). Their build methods are defined in file [data/\_\_init\_\_.py](./data/__init__.py). 

For some data, the official data transforms are not sufficient. The custom transforms can be placed in a new file.


### Log
Our framework now support basic and common log output formats, such as *.txt file*, *.yaml file*, *[wandb](https://docs.wandb.ai/)* and *tensorboard*. It is also very simple to use these formats for logging. Basically, you only need to pay attention to the related methods of the Metrics (in file [log/log.py](./log/log.py)) and Logger (in file [log/logger.py](./log/logger.py)) classes.


### Utils
In package [utils](./utils), there are some general methods that are not fall into the aforementioned categories. Like `is_distributed()` to determine whether it is in DDP mode, and so on.

[//]: # (## Project Tree)

[//]: # (```bash)

[//]: # (.)

[//]: # (├── configs   # 配置文件以及相关函数)

[//]: # (│   ├── __init__.py)

[//]: # (│   ├── resnet18_mnist.yaml)

[//]: # (│   └── utils.py)

[//]: # (├── data      # 数据操作，包括构造 Dataset DataLoader)

[//]: # (│   ├── __init__.py)

[//]: # (│   ├── mnist.py)

[//]: # (│   └── utils.py)

[//]: # (├── log       # 日志操作，包括了日志输出、存储和计算)

[//]: # (│   ├── __init__.py)

[//]: # (│   ├── logger.py)

[//]: # (│   └── log.py)

[//]: # (├── models    # 网络结构)

[//]: # (│   ├── __init__.py)

[//]: # (│   ├── resnet18.py)

[//]: # (│   └── utils.py)

[//]: # (├── utils     # utils)

[//]: # (│   ├── __init__.py)

[//]: # (│   └── utils.py)

[//]: # (├── LICENSE)

[//]: # (├── README.md)

[//]: # (├── engine.py)

[//]: # (├── main.py)

[//]: # (└── run.sh)

[//]: # (```)


## Config file

整个项目的配置信息（例如模型具体层数、训练方式、各类超参数等）都放在文件夹 [configs](./configs) 中。
其中配置信息以`.yaml`文件的形式存储，样例如下：
```yaml
MODE: # "train" or "eval", for the main.py script.
DEVICE: cuda
AVAILABLE_GPUS: 0,1,2,3,4,5,6,7
#GPUS: [0, 1, 2, 3, 4, 5, 6, 7]

# DATA:
DATASET: MNIST
DATA_PATH: ./dataset/MNIST/
NUM_CLASSES: 10
NUM_WORKERS: 2

# MODEL:
PRETRAINED:

# Train Setting:
SEED: 42
USE_DISTRIBUTED: False
LR: 0.0001
WEIGHT_DECAY: 0.0001
SCHEDULER_TYPE: MultiStep
SCHEDULER_MILESTONES: [3, ]
SCHEDULER_GAMMA: 0.5
BATCH_SIZE: 256
BATCH_SIZE_AVERAGE: True
EPOCHS: 5
RESUME_MODEL:
RESUME_OPTIMIZER: True
RESUME_SCHEDULER: True
RESUME_STATES: True

# Eval:
EVAL_MODEL:

# Outputs:
OUTPUTS_DIR: ./outputs/temp/
OUTPUTS_PER_STEP: 40
USE_TENSORBOARD: True
USE_WANDB: True
PROJECT_NAME: CV_Framework
EXP_NAME: default
EXP_GROUP:
```

