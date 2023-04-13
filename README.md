# README.md


This project builds a simple vision model training framework implemented by [PyTorch](https://pytorch.org/), has the following features:

- [x] Model training
- [x] Model evaluation
- [x] Logging to file / tensorboard / wandb
- [x] Distributed training and evaluation

In this project, the digit classification on MNIST is used as a toy task. You should download [MNIST dataset](https://github.com/HELLORPG/CV-Framework/releases/download/v1.0.0/MNIST.zip) and put it in the `./datasets/MNIST/` folder without decompression.


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


## Runtime setting

All runtime settings are recorded in a `.yaml` file like [configs/resnet18_mnist.yaml](./configs/resnet18_mnist.yaml).
In addition, some settings can be set by script parameters at runtime, such as `--batch-size`, `--lr`, and so on.

```yaml
MODE: # "train" or "eval", for the main.py script.
DEVICE: cuda
AVAILABLE_GPUS: 0,1,2,3,4,5,6,7

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

