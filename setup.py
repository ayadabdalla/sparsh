#!/usr/bin/env python

from setuptools import setup, find_packages

long_description = "SSL for tactile data"

setup(
    name="sparsh",
    version="0.0.1",
    author="Meta Research",
    description="SSL for tactile data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayadabdalla/sparsh",
    packages=find_packages(),  # Searches the current directory, where tactile_ssl is
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
