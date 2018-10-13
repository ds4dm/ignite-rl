# ignite-rl
This repository collects [Ignite](https://pytorch.org/ignite) engines for deep reinforcment learning algorithms.
Implementations run on GPU through [Pytorch](https://pytorch.org/).
This package does not aim to provide the most efficient implementations (although efficiency is a concern), but rather to showcase clean implementations and code reusability using Ignite.

Generally speaking, the user is expected to provide the model to be learn, as well as an [OpenAi Gym](https://gym.openai.com/) compatible environement.

## Installation
The project requires **Python 3.6+**.

To install, use [pipenv](pipenv.org), the [recomended](https://packaging.python.org/tutorials/managing-dependencies/) dependency manager for Python (also works with pip):
```
$ pipenv install git+https://github.com/ds4dm/ignite-rl.git#egg=irl
```
(not yet on [PyPI](https://pypi.org/))
