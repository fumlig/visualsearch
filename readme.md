<h1 align="center"> Learning to Search for Targets</h1>
<p align="center">A Deep Reinforcement Learning Approach to Visual Search in Unknown Environments</p>

## Abstract

## To do

Presentation:

- conclusion
- emphasize reasoning over scene appearance
- fix reinforcement learning theory

Questions:

- what to include?
- ppo description
- loss functions
- 

- Softmax temperature
- Visualize actions and scene
- Show attention
- Fix readme

## Reproduce

To reproduce results, run:

```bash
pip install -r requirements.txt

./run.sh
```

This will run all experiments that are presented in the report.
Learning information is written to the [results](./results) directory, checkpoints and the final model to [models](./models), and tensorboard logs to [logs](./logs).
