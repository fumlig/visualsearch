from agents import ac
from agents import ppo
from agents import cnn


ALGORITHMS = dict(
    ppo=ppo
)


def algorithm(alg_id):
    return ALGORITHMS.get(alg_id)