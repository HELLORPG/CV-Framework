# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Some utils.


import yaml


def yaml_to_dict(path: str):
    """
    Read a yaml file into a dict.

    Args:
        path (str): The path of yaml file.

    Returns:
        A dict.
    """
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


if __name__ == '__main__':
    config = yaml_to_dict("./configs/resnet18_mnist.yaml")


