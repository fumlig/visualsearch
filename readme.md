<h1 align="center"> Learning to Search for Targets</h1>
<p align="center">A Deep Reinforcement Learning Approach to Visual Search in Unknown Environments</p>

&lt;oskar dot lundin at pm dot me&gt;

## Abstract

Visual search is the perceptual task of locating a target in a visual environment. Due to
applications in areas like search and rescue, surveillance and home assistance, it is of great
interest to automate visual search. An autonomous system can potentially search more ef-
ficiently than a manually controlled one, and has the advantages of reduced risk and cost
of labour. However, manually designing search algorithms that properly utilize patterns
to search efficiently is not trivial. Different environments may exhibit vastly different pat-
terns, and visual cues may be difficult to pick up. A learning system has the advantage of
being applicable to any environment where there is a sufficient number of samples to learn
from.
In this thesis, we investigate how an agent that learns to search can be implemented
with deep reinforcement learning. Our approach jointly learns control of visual attention,
recognition and localization from a set of sample search scenarios. A recurrent convolu-
tional neural network takes an image of the visible region and the agent’s position as input.
Its outputs indicate whether a target is visible and control where the agent looks next. The
recurrent step serves as a memory that lets the agent utilize features of the explored envi-
ronment when searching. We compare two memory architectures: an LSTM, and a spatial
memory that remembers structured visual information. Through experimentation in three
simulated environments, we find that the spatial memory architecture achieves superior
search performance. It also searches more efficiently than a set of baselines that do not
utilize the appearance of the environment, and achieves similar performance to that of a
human searcher. Finally, the spatial memory scales to larger search spaces and is better at
generalizing from a limited number of training samples.

## To do

Report:

- discuss hyperparameter tuning
- proof-read
- feedback from jose, eric

Repository:

- document code
- scripts for experiments
- fix permanent link in report (https://zenodo.org/?)

## Overview

- [gym_search](./gym_search): Gym environments for visual search.
- [rl](./rl): Reinforcement learning library (agents, models and algorithms).

- [results](./results): Files with result metrics.
- [videos](./videos): Videos of agent behaviors.

- [report](./report): Thesis report.
- [slides](./slides): Thesis slides.

## Training & Testing

To train an agent, run:

```bash
./train.py <environment>

## Reproduce

To reproduce results, run:

```bash
pip install -r requirements.txt

./run.sh
```

This will run all experiments that are presented in the report.
Learning information is written to the [results](./results) directory, checkpoints and the final model to [models](./models), and tensorboard logs to [logs](./logs). Plotting is done in the [plot.ipynb](./plot.ipynb) notebook.
