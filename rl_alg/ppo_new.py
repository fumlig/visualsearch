import argparse
import datetime
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import gym_search


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

        # todo: shared?

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--videos", type=str, default="videos")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")

    # hyperparameters 
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--num-envs", type=int, default=8, help="parallell environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="steps per environment per rollout")
    parser.add_argument("--num-epochs", type=int, default=10, help="epochs when optimizing surrogate loss")
    parser.add_argument("--num-minibatches", type=int, default=64, help="number of mini-batches")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="lambda for general advantage estimation")

    parser.add_argument("--clip-range", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-range-vf", type=float, default=None, help="clipping parameter for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm for gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    
    args = parser.parse_args()

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches

    name = args.name if args.name else f"{args.env}-ppo-{datetime.datetime.now().isoformat()}"
    
    writer = SummaryWriter(f"logs/{name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    def make_env(id, idx, record, name, videos="videos"):
        env = gym.make(id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"{videos}/{name}")
        return env

    envs = gym.vector.SyncVectorEnv(
        [lambda: make_env(args.env, i, args.record, name, args.videos) for i in range(args.num_envs)]
    )

    # TRY NOT TO MODIFY: seeding

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        envs.seed(args.seed)

    torch.backends.cudnn.deterministic = args.deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5) # common epsilon



    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    num_timesteps = 0

    while num_timesteps < args.total_timesteps:

        # collect rollouts
        for step in range(0, args.num_steps):
            num_timesteps += 1 * args.num_envs
            
            obs[step] = next_obs
            dones[step] = next_done

            # select action and value
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # execute action
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # update info
            for item in info:
                if "episode" in item.keys():
                    print(f"num_timesteps={num_timesteps}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], num_timesteps)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], num_timesteps)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # train policy and value networks
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.num_epochs):

            # shuffle batches
            np.random.shuffle(b_inds)
            
            # train on batches
            for start in range(0, args.batch_size, args.minibatch_size):
                
                # select minibatches
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_range).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                # normaliize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_range_vf is not None:
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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], num_timesteps)
        writer.add_scalar("losses/value_loss", v_loss.item(), num_timesteps)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), num_timesteps)
        writer.add_scalar("losses/entropy", entropy_loss.item(), num_timesteps)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), num_timesteps)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), num_timesteps)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), num_timesteps)
        writer.add_scalar("losses/explained_variance", explained_var, num_timesteps)
        writer.add_scalar("charts/steps_per_second", int(num_timesteps / (time.time() - start_time)), num_timesteps)

        print(f"steps_per_second={int(num_timesteps / (time.time() - start_time))}")


    envs.close()
    writer.close()