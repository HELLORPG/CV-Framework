# @Author       : Ruopeng Gao
# @Date         : 2022/7/6
# @Description  : Data operators, such as data read, dataset, dataloader.

"""
In this package, should include the main operators about dataset.
Mainly include below parts:
1. Use dataset API (For example, COCO API) or just simple code to read files.
2. Build a PyTorch Dataset.
3. Build a PyTorch DataLoader.
4. Maybe you should design a PyTorch Sampler class.
5. Probably you should design how to transform the data.
6. Sometimes, you should build a function 'collate_fn' to tell the dataloader how to aggregate a batch of data.

The above features can be achieved in a single .py file or multi of them.
"""