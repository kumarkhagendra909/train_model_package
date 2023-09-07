from setuptools import setup, find_packages
setup(
    name="imageclassification",
    version="1.0.1",
    packages=find_packages(),
    description="A simple image classification deep learning model package which help in training the deep learning model",
    author="Khagedra Kumar Mandal",
    install_requires=[
        # Add any dependencies here.
        "torch>=2.0.1",
        "matplotlib>=3.7.1",
        "torchvision>=0.15.2",
        "torchinfo>=1.8.0",
        "tqdm>=4.65.0",
        # these dependencies are required to run this pip package on locakl 
        # since this package dont need to use any particular version of library or any library support
        # "torch"
        # "matplotlib"
        # "torchvision"
        # "torchinfo"
        # "os"
        # "tqdm"
    ],
)
