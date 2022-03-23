from agents import ppo
from agents import cnn
from agents.agents import Agent, SearchAgent, MemoryAgent


ALGORITHMS = {
    "ppo": ppo
}

AGENTS = {
    "search": SearchAgent,
}


def algorithm(algorithm_id):
    return ALGORITHMS.get(algorithm_id)


def agent(agent_id):
    return AGENTS.get(agent_id)