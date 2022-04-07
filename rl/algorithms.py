from imp import init_builtin
import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque


def proximal_policy_optimization(
    tot_timesteps,
    envs,
    agent,
    device,
    writer,
    learning_rate=2.5e-4,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    num_epochs=4,
    norm_adv=True,
    clip_range=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
):
    num_envs = envs.num_envs
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_batches = tot_timesteps // batch_size

    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    assert isinstance(envs.single_observation_space, gym.spaces.Dict)
    assert num_envs % num_minibatches == 0

    #hparams = locals()
    #hparams.pop(envs)
    #hparams.pop()

    #writer.add_text(
    #    "hyperparameters",
    #    "|param|value|\n|-|-|\n" +
    #    "\n".join([f"|{key}|{value}|" for key, value in args.hparams.items()]) 
    #)

    timestep = 0
    agent.to(device)
    optimizer = th.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    pbar = tqdm(total=tot_timesteps)
    ep_infos = deque(maxlen=num_envs)

    obss = {key: th.zeros((num_steps, num_envs) + space.shape).to(device) for key, space in envs.single_observation_space.items()}
    acts = th.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    rews = th.zeros((num_steps, num_envs)).to(device)
    dones = th.zeros((num_steps, num_envs)).to(device)
    logprobs = th.zeros((num_steps, num_envs)).to(device)
    vals = th.zeros((num_steps, num_envs)).to(device)

    obs = {key: th.tensor(o, dtype=th.float).to(device) for key, o in envs.reset().items()}
    done = th.zeros(num_envs).to(device)
    state = [s.to(device) for s in agent.initial(num_envs)]

    for _b in range(num_batches):

        # for train
        initial_state = [s.clone() for s in state]
        
        # rollout steps
        for step in range(num_steps):
            timestep += num_envs

            with th.no_grad():
                pi, v, state = agent(obs, state, done=done)
                val = v.flatten()
                act = pi.sample()
                logprob = pi.log_prob(act)
            
            next_obs, rew, next_done, infos = envs.step(act.cpu().numpy())
            rew = th.tensor(rew).to(device).view(-1)

            for key, observation in obs.items():
                obss[key][step] = observation

            rews[step] = rew
            dones[step] = done
            acts[step] = act
            logprobs[step] = logprob
            vals[step] = val

            obs = {key: th.tensor(o, dtype=th.float).to(device) for key, o in next_obs.items()}
            done = th.tensor(next_done, dtype=th.float).to(device)

            for i, info in enumerate(infos):
                if "episode" in info:
                    assert(done[i] == 1)

                    ep_info = info["episode"]
                    writer.add_scalar("episode/return", ep_info["r"], timestep)
                    writer.add_scalar("episode/length",  ep_info["l"], timestep)
                    
                    ep_infos.append(ep_info)

                    if "counters" in info:
                        for key, count in info["counters"].items():
                            writer.add_scalar(f"counter/{key}", count, timestep)


        # bootstrap value
        with th.no_grad():
            _, next_val, _ = agent(obs, state, done=done)
            advs = th.zeros_like(rews).to(device)
            last_gae_lambda = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_val = next_val.reshape(1, -1)
                    next_nonterminal = 1.0 - done
                else:
                    next_val = vals[t+1]
                    next_nonterminal = 1.0 - dones[t+1]

                delta = rews[t] + gamma*next_val*next_nonterminal - vals[t]
                last_gae_lambda = delta + gamma*gae_lambda*next_nonterminal*last_gae_lambda
                advs[t] = last_gae_lambda

            rets = advs + vals


        # train policy
        b_obss = {key: o.reshape((-1,) + envs.single_observation_space[key].shape) for key, o in obss.items()}
        b_acts = acts.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape((-1))
        b_logprobs = logprobs.reshape((-1,))
        b_vals = vals.reshape((-1,))
        b_advs = advs.reshape((-1,))
        b_rets = rets.reshape((-1,))

        envs_idx = np.arange(num_envs)
        flat_idx = np.arange(batch_size).reshape(num_steps, num_envs)

        clip_fracs = []

        # batch contains one for each step and env

        for _epoch in range(num_epochs):
            np.random.shuffle(envs_idx)

            # each minibatch contains some of the environments
            for mb_begin in range(0, num_envs, num_envs // num_minibatches):
                mb_end = mb_begin + num_envs // num_minibatches
                mb_envs_idx = envs_idx[mb_begin:mb_end]
                mb_idx = flat_idx[:, mb_envs_idx].ravel()
                
                mb_obss = {key: o[mb_idx] for key, o in b_obss.items()}
                mb_acts = b_acts[mb_idx]
                mb_dones = b_dones[mb_idx]
                mb_logprobs = b_logprobs[mb_idx]
                mb_vals = b_vals[mb_idx]
                mb_advs = b_advs[mb_idx]
                mb_rets = b_rets[mb_idx]
                mb_state = [s[mb_envs_idx] for s in initial_state]

                pi, v, _ = agent(mb_obss, mb_state, done=mb_dones)
                new_val = v.view(-1)
                act = mb_acts.long()
                new_logprob = pi.log_prob(act)
                entropy = pi.entropy()
                logratio = new_logprob - mb_logprobs
                ratio = logratio.exp()

                with th.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > clip_range).float().mean().item())

                if norm_adv:
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                
                # policy loss
                pg_loss1 = -mb_advs * ratio
                pg_loss2 = -mb_advs * th.clamp(ratio, 1-clip_range, 1+clip_range)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # value loss
                if clip_vloss:
                    val_loss_unclipped = (new_val - mb_rets)**2
                    val_clipped = mb_vals + th.clamp(new_val-mb_vals, -clip_range, clip_range)
                    val_loss_clipped = (val_clipped - mb_rets)**2
                    val_loss_max = th.max(val_loss_unclipped, val_loss_clipped)
                    val_loss = 0.5 * val_loss_max.mean()
                else:
                    val_loss = 0.5 * ((new_val - mb_rets)**2).mean()
                
                ent_loss = entropy.mean()
                
                loss = pg_loss - ent_coef*ent_loss + vf_coef*val_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
            
            # early break
            if target_kl is not None and approx_kl > target_kl:
                break
        
        writer.add_scalar("losses/value_loss", val_loss.item(), timestep)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), timestep)
        writer.add_scalar("losses/entropy_loss", ent_loss.item(), timestep)

        pbar.update(timestep - pbar.n)

        if ep_infos:
            avg_ret = np.mean([ep_info["r"] for ep_info in ep_infos])
            avg_len = np.mean([ep_info["l"] for ep_info in ep_infos])
            pbar.set_description(f"ret {round(avg_ret)}, len {round(avg_len)}")



def deep_q_network(
    tot_timesteps,
    envs,
    agent,
    device,
    writer,
    learning_rate=2.5e-4,
    buffer_size=10000,
    gamma=0.99,
    target_net_freq=500,
    max_grad_norm=0.5,
    batch_size=32,
    start_eps=1,
    end_eps=0.05,
    exploration_frac=0.8,
    learning_start=10000,
    train_freq=1,
):
    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    num_envs = envs.num_envs

    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    assert isinstance(envs.single_observation_space, gym.spaces.Dict)

    agent.to(device)
    optimizer = th.optim.Adam(agent.parameters(), lr=learning_rate)
    target_net = type(agent)(envs).to(device)
    target_net.load_state_dict(agent.state_dict())

    buf_obss = {key: th.zeros((buffer_size, num_envs) + space.shape).to(device) for key, space in envs.single_observation_space.items()}
    buf_acts = th.zeros((buffer_size, num_envs) + envs.single_action_space.shape).to(device)
    buf_rews = th.zeros((buffer_size, num_envs)).to(device)
    buf_dones = th.zeros((buffer_size, num_envs)).to(device)

    buf_next_obss = {key: th.zeros((buffer_size, num_envs) + space.shape).to(device) for key, space in envs.single_observation_space.items()}
    buf_next_dones = {key: th.zeros((buffer_size, num_envs) + space.shape).to(device) for key, space in envs.single_observation_space.items()}

    buf_pos = 0
    buf_full = False

    obs = {key: th.tensor(o).to(device) for key, o in envs.reset().items()}
    dones = th.zeros(num_envs).to(device)
    states = [s.to(device) for s in agent.initial(num_envs)]

    for timestep in tqdm(range(tot_timesteps)):

        initial_states = [s.clone() for s in states]

        epsilon = linear_schedule(start_eps, end_eps, exploration_frac * tot_timesteps, timestep)
        if random.random() < epsilon:
            actions = th.tensor([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            pi, _, states = agent(obs, states, done=dones)
            actions = th.argmax(pi.probs, dim=1)

        next_obs, rewards, dones, infos = envs.step(actions)
        next_obs = {key: th.tensor(o).to(device) for key, o in next_obs.items()}
        rewards = th.tensor(rewards).to(device)
        dones = th.tensor(dones, dtype=th.float).to(device)

        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                writer.add_scalar("charts/episode_return", ep_info["r"], timestep)
                writer.add_scalar("charts/episode_length",  ep_info["l"], timestep)
                writer.add_scalar("charts/epsilon", epsilon, timestep)

        #real_next_obs = {key: o.clone() for key, o in next_obs.items()}
        #for idx, d in enumerate(dones):
        #    if d:
        #        real_next_obs = {key: o[idx] for key, o in next_obs.items()}
        #        real_next_obs[idx] = info[idx]["terminal_observation"]

        # add to buffer
        for key in obs.keys():
            buf_obss[key][buf_pos] = obs[key]
            buf_next_obss[key][buf_pos] = obs[key].clone()
        
        buf_acts[buf_pos] = actions.clone()
        buf_rews[buf_pos] = rewards.clone()
        buf_dones[buf_pos] = dones.clone()

        buf_pos += 1
        if buf_pos == buffer_size:
            buf_full = True
            buf_pos = 0

        obs = next_obs

        if timestep > learning_start and timestep % train_freq == 0:

            # sample from buffer
            upper_bound = buffer_size if buf_full else buf_pos
            batch_idx = np.random.randint(0, upper_bound, size=batch_size)
            env_idx = np.random.randint(0, high=num_envs, size=(len(batch_idx),))
            
            obss = {key: obs[batch_idx, env_idx].to(device) for key, obs in buf_obss.items()} # normalize?
            next_obss = {key: obs[batch_idx, env_idx].to(device) for key, obs in buf_next_obss.items()}
            acts = buf_acts[batch_idx, env_idx].to(device)
            dones = buf_dones[batch_idx, env_idx]
            rews = buf_rews[batch_idx, env_idx] # normalize?

            with th.no_grad():
                pi, _, _, = target_net(next_obss, state, done)
                target_max, _ = target_net(next_obss).max(dim=1)
                td_target = rews.flatten() + gamma * target_max * (1 - dones.flatten())
            pi, _, _ = agent(obss, initial_states[:, env_idx])
            old_val = pi.gather(1, acts).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if timestep % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, timestep)
                writer.add_scalar("losses/q_values", old_val.mean().item(), timestep)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(agent.parameters()), max_grad_norm)
            optimizer.step()

            # update the target network
            if timestep % target_net_freq == 0:
                target_net.load_state_dict(agent.state_dict())

    envs.close()
    writer.close()


ALGORITHMS = {
    "ppo": proximal_policy_optimization,
    "dqn": deep_q_network
}


def learn(id, tot_timesteps, envs, agent, device, writer, **kwargs):
    return ALGORITHMS.get(id)(tot_timesteps, envs, agent, device, writer, **kwargs)
