import torch as th
import gym
import gym_search

from torch.utils.tensorboard import SummaryWriter
from agents.ac import ActorCritic
from agents.ppo import learn

if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    envs = gym.vector.make(
        "SearchSparse-v0",
        64,
        asynchronous=True,
        wrappers=[gym.wrappers.FlattenObservation, gym.wrappers.RecordEpisodeStatistics]
    )
    envs = gym.wrappers.NormalizeReward(envs)

    agent = ActorCritic(envs).to(device)

    writer = SummaryWriter("logs/test")

    learn(envs, agent, device, writer, tot_timesteps=int(100e6))

    envs.close()
    writer.close()