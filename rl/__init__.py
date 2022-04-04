import rl.algorithms
import rl.agents


ALGORITHMS = {
    "ppo": rl.algorithms.ProximalPolicyOptimization,
    "dqn": rl.algorithms.DeepQNetworks
}

AGENTS = {
    "random": rl.agents.RandomAgent,
    # baselines
    "image": rl.agents.ImageAgent,
    "recurrent": rl.agents.RecurrentAgent,
    "baseline": rl.agents.BaselineAgent,
    # our
    "map": rl.agents.MapAgent
}


def algorithm(algorithm_id):
    return ALGORITHMS.get(algorithm_id)

def agent(agent_id):
    return AGENTS.get(agent_id)