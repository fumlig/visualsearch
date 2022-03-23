from rl.algorithms import ProximalPolicyOptimization
from rl.agents import RandomAgent, SearchAgent, MemoryAgent, ImpalaAgent


ALGORITHMS = {
    "ppo": ProximalPolicyOptimization
}

AGENTS = {
    "our": SearchAgent,
    "random": RandomAgent,
    "impala": ImpalaAgent,
    "memory": MemoryAgent,
}


def algorithm(algorithm_id):
    return ALGORITHMS.get(algorithm_id)

def agent(agent_id):
    return AGENTS.get(agent_id)