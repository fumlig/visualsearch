<h1 align="center"> Learning to Search for Targets</h1>
<p align="center">A Deep Reinforcement Learning Approach to Visual Search in Unknown Environments</p>

## Abstract

todo

## To do

- Abstract
- Conclusion
- Source criticism
- Wider context
- Send to Fredrik

- Learning curves
- ~Update search paths~
- Illustrate memory
- Update camera metrics
- Softmax temperature
- Visualize actions and scene
- Show attention
- Update presentation
- Dumb baseline
- Make it clear that observation is limited


## Reproduce

To reproduce results, run:

```bash
pip install -r requirements.txt

./run.sh 
```

This will run all experiments that are presented in the report.
Learning information is written to the [results](./results) directory, checkpoints and the final model to [models](./models), and tensorboard logs to [logs](./logs).