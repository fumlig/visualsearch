# Master's Thesis

## abstract

## experiments

Run all experiments:

```bash
pip install -e .
tensorboard --logdir=logs --samples_per_plugin=scalars=10000 &

python3 train.py 
```