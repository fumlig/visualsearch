from rl.algorithms import ProximalPolicyOptimization
from rl.agents import Agent, SearchAgent, MemoryAgent, KingAgent


ALGORITHMS = {
    "ppo": ProximalPolicyOptimization
}

AGENTS = {
    "our": SearchAgent,
}


def algorithm(algorithm_id):
    return ALGORITHMS.get(algorithm_id)

def agent(agent_id):
    return AGENTS.get(agent_id)