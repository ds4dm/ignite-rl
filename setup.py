# coding: utf-8

import sys
from setuptools import setup, find_packages


needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []


setup(
    name="ignite-rl",
    version="0.1",
    packages=find_packages(),
    author="Antoine Prouvost",
    license="MIT",
    url="github.com/ds4dm/ignite-rl",
    description="Collection of ignite engines for reinforcment learning.",
    long_description=open("README.md").read(),
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "pytorch-ignite",
        "attrs"
    ],
    # Testing
    # usage: python setup.py pytest --addopts --cov=irl
    setup_requires=[] + pytest_runner,
    tests_require=[
        "pytest",
        "pytest-cov",
        "mock",
        "gym"
    ]
)
