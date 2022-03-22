from agents import ppo
from agents import cnn
from agents.agents import Agent, SearchAgent, MemoryAgent

ALGORITHMS = dict(
    ppo=ppo
)


def algorithm(alg_id):
    return ALGORITHMS.get(alg_id)

