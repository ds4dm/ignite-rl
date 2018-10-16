# coding: utf-8

from setuptools import setup, find_packages


setup(
    name="Ignite-Rl",
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
    ]
)
