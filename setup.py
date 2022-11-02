from setuptools import setup
from setuptools import find_packages
import os
import re

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="frgtrainview",
    version="0.1",
    description="Automated analsyis of the Fenning Research Group automated synthesis platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Deniz N. Cakan",
    author_email="dcakan@eng.ucsd.edu",
    download_url="https://github.com/fenning-research-group/PASCAL_TrainView/batch_process",
    license="MIT",
    install_requires=[
        "numpy",
        "aiohttp",
        "pyyaml",
        "scipy",
        "mixsol",
        "ortools",
        "matplotlib",
        "pandas",
        "pyserial",
        "natsort",
        "ntplib",
        "websockets",
        "dill",
        "PyQt5",
        "tifffile",
        "scikit-image",
        "ax-platform",
        "ntplib",
    ],
    packages=find_packages(),
    package_data={
        "": [
            "hardware/*.yaml",
            "hardware/*/*.yaml",
            "hardware/*/*/*.yaml",
            "hardware/*/*/*.json",
            "Examples/*.ipynb",
        ],
    },
    include_package_data=True,
    keywords=["materials", "science", "machine", "automation", "data science"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # entry_points={
    #     'console_scripts': [
    #         'meg = megnet.cli.meg:main',
    #     ]
    # }
)
