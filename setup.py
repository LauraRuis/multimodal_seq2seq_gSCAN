from setuptools import find_packages
from setuptools import setup

setup(name='MultimodalSeq2seqGscan',
      version='0.1',
      url='https://github.com/LauraRuis/multimodal_seq2seq_gSCAN',
      author='Laura Ruis',
      author_email='lauraruis@live.nl',
      packages=find_packages(include=['seq2seq']),
      install_requires=[
            'imageio~=2.9.0',
            'setuptools~=49.2.1',
            'pronounceable~=0.1.3',
            'PyQt5~=5.15.4',
            'opencv-python~=4.5.1.48',
            'xlwt~=1.3.0',
            'torch~=1.8.0',
            'torchvision~=0.9.0',
            'numpy~=1.20.1',
            'gym~=0.18.0',
            'matplotlib~=3.3.4'
      ]
)