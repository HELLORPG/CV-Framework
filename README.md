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


## Project Tree
```bash
.
├── configs
│   ├── __init__.py
│   ├── resnet18_mnist.yaml
│   └── utils.py
├── data
│   ├── __init__.py
│   ├── mnist.py
│   └── utils.py
├── log
│   ├── __init__.py
│   ├── logger.py
│   └── log.py
├── models
│   ├── __init__.py
│   ├── resnet18.py
│   └── utils.py
├── utils
│   ├── __init__.py
│   └── utils.py
├── LICENSE
├── README.md
├── engine.py
├── main.py
└── run.sh
```
