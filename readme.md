<h1 align="center"> Learning to Search for Targets</h1>
<p align="center">A Deep Reinforcement Learning Approach to Visual Search in Unknown Environments</p>

## Abstract

## To do

Report:

- discuss hyperparameter tuning
- proof-read

Presentation:

- rework conclusion
- fix animations

Repository:

- document code
- scripts for experiments
- fix permanent link in report (https://zenodo.org/?)

Other:

- talk to fredrik about future work, evaluation of time at FOI

## Reproduce

To reproduce results, run:

```bash
pip install -r requirements.txt

./run.sh
```

This will run all experiments that are presented in the report.
Learning information is written to the [results](./results) directory, checkpoints and the final model to [models](./models), and tensorboard logs to [logs](./logs).
