from rl.algorithms import ProximalPolicyOptimization
from rl.agents import Agent, SearchAgent, MemoryAgent, ImpalaAgent


ALGORITHMS = {
    "ppo": ProximalPolicyOptimization
}

AGENTS = {
    "our": SearchAgent,
    "impala": ImpalaAgent,
    "memory": MemoryAgent
}


def algorithm(algorithm_id):
    return ALGORITHMS.get(algorithm_id)

def agent(agent_id):
    return AGENTS.get(agent_id)