from agents import ac
from agents import ppo
from agents import cnn


def algorithm(alg_id):
    return dict(
        ppo=ppo
    ).get(alg_id)