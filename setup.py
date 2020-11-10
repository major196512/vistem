# python -m pip install -e pytorch.template

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

import os
import glob
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 5], "Requires PyTorch >= 1.5"


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "vistem", "extensions")

    main_source = os.path.join(extensions_dir, "wrapper.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"))

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and (CUDA_HOME is not None)) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "vistem.ext",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

setup(
    name="vistem",
    version="0.1.1",
    author="Taeho Choi",
    url="https://github.com/major196512/vistem",
    packages=find_packages(exclude=("configs")),
    python_requires=">=3.6",
    install_requires=[
        "termcolor>=1.1",
        "Pillow>=6.0",
        "yacs>=0.1.6",
        "tabulate>=0.8",
        "opencv-python>=4.4.0.42",
        "pycocotools>=2.0.2",
        "cloudpickle>=1.6.0",
        "wandb>=0.10.8"
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension}
)
