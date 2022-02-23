#!/usr/bin/env python3

import argparse
import os
import random
import time
import datetime
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from tqdm import tqdm

import gym
import gym_search


def make_env(id, seed, idx):
    def thunk():
        env = gym.make(id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

        num_features = gym.spaces.flatdim(envs.single_observation_space)

        self.network = nn.Identity()

        #self.network = nn.Sequential(
        #    layer_init(nn.Linear(num_features, 256)),
        #    nn.Tanh(),
        #    layer_init(nn.Linear(256, 64)),
        #    nn.Tanh(),
        #)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(num_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )

    def forward(self, x):
        y = self.network(x)
        pi = self.policy(y)
        vf = self.value(y)
        return pi, vf

    def policy(self, y):
        logits = self.actor(y)
        pi = Categorical(logits=logits)
        return pi

    def value(self, y):
        vf = self.critic(y)
        return vf


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="id of gym environment")
    parser.add_argument("--name", type=str, default="ppo", help="name of experiment")

    parser.add_argument("--learning-rate", type=float, default=5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--num-timesteps", type=int, default=int(10e6), help="total number of timesteps")
    parser.add_argument("--deterministic", action="store_true", help="make torch deterministic")

    parser.add_argument("--num-envs", type=int, default=64, help="number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=64, help="number of steps to run in each environment per policy rollout") # per the PPO paper it is possible that this should be "much smaller than episode length"
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, help="toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.999, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-range", type=float, default=0.2, help="the surrogate clipping range")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.") # todo: only for shared
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy") # when is an entropy coefficient of zero good?
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    name = f"{args.env.lower()}-{args.name}-{datetime.datetime.now().isoformat()}"

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env, args.seed + i, i) for i in range(args.num_envs)]
    )
    #envs = gym.wrappers.NormalizeReward(envs) # todo: not sure where this should be
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    writer = SummaryWriter(f"logs/{name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # todo
    #writer.add_graph(
    #    agent,
    #    torch.Tensor(envs.observation_space.sample()).to(device).float(),
    #)

    #writer.add_video(
    #    "videos/test",
    #    torch.rand((1, 16, 3, 10, 10)),
    #    global_step=0,
    #    fps=4
    #)


    # buffer
    buf_shape = (args.num_steps, args.num_envs)

    obs = torch.zeros(buf_shape + envs.single_observation_space.shape).to(device)
    actions = torch.zeros(buf_shape + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros(buf_shape).to(device)
    rewards = torch.zeros(buf_shape).to(device)
    dones = torch.zeros(buf_shape).to(device)
    values = torch.zeros(buf_shape).to(device)

    # initialize
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.num_timesteps // args.batch_size

    pbar = tqdm(total=args.num_timesteps)

    for update in range(1, num_updates + 1):
        # lr annealing (todo: use pytorch builtins?)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # rollout
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            obs[step] = next_obs
            dones[step] = next_done

            # query agent
            with torch.no_grad():
                pi, value = agent(next_obs)
                action = pi.sample()
                logprob = pi.log_prob(action)
            
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # step environment
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # store finished episodes
            for i in info:
                if "episode" in i:
                    episode_r = i["episode"]["r"]
                    episode_l = i["episode"]["l"]

                    writer.add_scalar("charts/ep_rew", episode_r, global_step)
                    writer.add_scalar("charts/ep_len", episode_l, global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            _, next_value = agent(next_obs)
            next_value = next_value.reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # todo: make rollouts cleaner

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # update policy on batch
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # random batch order
            np.random.shuffle(b_inds)

            # do it in minibatches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                pi, newvalue = agent(b_obs[mb_inds])
                action = b_actions.long()[mb_inds]
                newlogprob = pi.log_prob(action)
                entropy = pi.entropy()

                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_range).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_range,
                        args.clip_range,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # aggregate loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # break early
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # record statistics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        step_rate = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/step_rate", step_rate, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        pbar.update(global_step - pbar.n)

    envs.close()
    writer.close()
    pbar.close()

    torch.save(agent, f"models/{name}.pt")
