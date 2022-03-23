import rl.algorithms
import rl.agents


ALGORITHMS = {
    "ppo": rl.algorithms.ProximalPolicyOptimization
}

AGENTS = {
    "random": rl.agents.RandomAgent,
    # baselines
    "image": rl.agents.ImageAgent,
    "memory": rl.agents.MemoryAgent,
    # our
    "our": rl.agents.SearchAgent
}


def algorithm(algorithm_id):
    return ALGORITHMS.get(algorithm_id)

def agent(agent_id):
    return AGENTS.get(agent_id)