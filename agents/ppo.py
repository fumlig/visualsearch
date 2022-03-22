import gym
import numpy as np
import torch as th
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque


def learn(
    tot_timesteps,
    envs,
    agent,
    device,
    writer=None,
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
    target_kl=None
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

    agent.to(device)
    optimizer = th.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obss = {key: th.zeros((num_steps, num_envs) + space.shape).to(device) for key, space in envs.single_observation_space.items()}
    rews = th.zeros((num_steps, num_envs)).to(device)
    dones = th.zeros((num_steps, num_envs)).to(device)
    acts = th.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = th.zeros((num_steps, num_envs)).to(device)
    vals = th.zeros((num_steps, num_envs)).to(device)

    obs = {key: th.tensor(observation, dtype=th.float).to(device) for key, observation in envs.reset().items()}
    done = th.zeros(envs.num_envs).to(device)
    state = agent.initial(num_envs)

    pbar = tqdm(total=tot_timesteps)
    ep_infos = deque(maxlen=100)

    timestep = 0

    for _batch in range(num_batches):
        # for train
        initial_state = (state[0].clone(), state[1].clone())
        
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

            obs = {key: th.tensor(observation, dtype=th.float).to(device) for key, observation in next_obs.items()}
            done = th.tensor(next_done, dtype=th.float).to(device)

            for i, info in enumerate(infos):
                if "episode" in info:
                    assert(done[i] == 1)

                    ep_info = info["episode"]
                    writer.add_scalar("charts/episode_return", ep_info["r"], timestep)
                    writer.add_scalar("charts/episode_length",  ep_info["l"], timestep)
                    
                    ep_infos.append(ep_info)
                    #writer.add_scalar("charts/average_return", np.mean([ep_info["r"] for ep_info in ep_infos]), timestep)
                    #writer.add_scalar("charts/average_length", np.mean([ep_info["l"] for ep_info in ep_infos]), timestep)


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
        b_obss = {key: observation.reshape((-1,) + envs.single_observation_space[key].shape) for key, observation in obss.items()}
        b_acts = acts.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape((-1))
        b_logprobs = logprobs.reshape((-1,))
        b_vals = vals.reshape((-1,))
        b_advs = advs.reshape((-1,))
        b_rets = rets.reshape((-1,))

        envs_idx = np.arange(num_envs)
        flat_idx = np.arange(batch_size).reshape(num_steps, num_envs)

        clip_fracs = []

        for _epoch in range(num_epochs):
            np.random.shuffle(envs_idx)

            for mb_begin in range(0, num_envs, num_envs // num_minibatches):
                mb_end = mb_begin + num_envs // num_minibatches
                mb_envs_idx = envs_idx[mb_begin:mb_end]
                mb_idx = flat_idx[:, mb_envs_idx].ravel()
                
                mb_obss = {key: b_observation[mb_idx] for key, b_observation in b_obss.items()}
                mb_acts = b_acts[mb_idx]
                mb_dones = b_dones[mb_idx]
                mb_logprobs = b_logprobs[mb_idx]
                mb_vals = b_vals[mb_idx]
                mb_advs = b_advs[mb_idx]
                mb_rets = b_rets[mb_idx]
                mb_state = (initial_state[0][:, mb_envs_idx], initial_state[1][:, mb_envs_idx])

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
                    val_clipped = mb_vals + th.clamp(new_val-mb_vals, -clip_range, clip_range) # todo: this clipping is different (not added with 1)
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

        # write summary
        #pred_vals = batch_vals.cpu().numpy()
        #true_vals = batch_rets.cpu().numpy()
        
        writer.add_scalar("losses/value_loss", val_loss.item(), timestep)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), timestep)
        writer.add_scalar("losses/entropy_loss", ent_loss.item(), timestep)

        pbar.update(timestep - pbar.n)
