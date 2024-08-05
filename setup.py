import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

def convert_to_absolute_path(path):
    # do like in bash - pushd to this file's directory, find the absolute path, popd
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.abspath(path)
    os.chdir(cwd)
    return abs_path

NAME = "Contrastive Loss"
VERSION = "0.1"
DESCRIPTION = "Contrastive Loss: Accelerated C++ PyTorch Extension for Contrastive Loss"
LONG_DESCRIPTION = "Contrastive Loss: Accelerated C++ PyTorch Extension for Contrastive Loss"
URL = "None"
AUTHOR = "Amit Abecasis"
AUTHOR_EMAIL = "amitabe@weizmann.ac.il"
LICENSE = "MIT License"

SOURCE_FILES = [
    "src/snnl.cpp",
    "src/distances.cpp",
    "src/utilities.cpp",
]

INCLUDE_DIRS = [
    "./include",
]
INCLUDE_DIRS = [convert_to_absolute_path(path) for path in INCLUDE_DIRS]

REQUIRED_PACKAGES = [
    "torch",
]

EXTRA_PACKAGES = {
}

ext_modules = [
    CppExtension(
        name="contrastive",
        sources=SOURCE_FILES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=['-O3'],
    ),
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

