# python -m pip install -e pytorch.template

from setuptools import find_packages, setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 5], "Requires PyTorch >= 1.5"


setup(
    name="vistem",
    version="0.0.beta",
    author="major196512",
    # url="https://github.com/facebookresearch/detectron2",
    packages=find_packages(exclude=("configs")),
    python_requires=">=3.6",
    install_requires=[
        "termcolor>=1.1",
        "Pillow>=6.0",
        "yacs>=0.1.6",
    ],
)
