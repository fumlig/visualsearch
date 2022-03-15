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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    assert isinstance(envs.single_observation_space, gym.spaces.Dict)

    num_envs = envs.num_envs
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_batches = tot_timesteps // batch_size

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

    pbar = tqdm(total=tot_timesteps)
    ep_infos = deque(maxlen=100)

    timestep = 0

    for _batch in range(num_batches):
        # rollout steps
        for step in range(num_steps):
            timestep += num_envs

            with th.no_grad():
                pi, v = agent(obs)
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
                    writer.add_scalar("charts/average_return", np.mean([ep_info["r"] for ep_info in ep_infos]), timestep)
                    writer.add_scalar("charts/average_length", np.mean([ep_info["l"] for ep_info in ep_infos]), timestep)


        # bootstrap value
        with th.no_grad():
            _, next_val = agent(obs) # todo: use or ignore state?
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
        batch_idx = np.arange(batch_size)

        batch_obss = {key: observation.reshape((-1,) + envs.single_observation_space[key].shape) for key, observation in obss.items()}
        batch_acts = acts.reshape((-1,) + envs.single_action_space.shape)
        batch_logprobs = logprobs.reshape((-1,))
        batch_vals = vals.reshape((-1,))
        batch_advs = advs.reshape((-1,))
        batch_rets = rets.reshape((-1,))

        clip_fracs = []

        for _epoch in range(num_epochs):
            np.random.shuffle(batch_idx)

            for minibatch in range(num_minibatches):
                minibatch_idx = batch_idx[minibatch*minibatch_size:(minibatch+1)*minibatch_size]
                
                minibatch_obss = {key: batch_observation[minibatch_idx] for key, batch_observation in batch_obss.items()}
                minibatch_acts = batch_acts[minibatch_idx]
                minibatch_logprobs = batch_logprobs[minibatch_idx]
                minibatch_vals = batch_vals[minibatch_idx]
                minibatch_advs = batch_advs[minibatch_idx]
                minibatch_rets = batch_rets[minibatch_idx]

                pi, v = agent(minibatch_obss)
                new_val = v.view(-1)
                act = minibatch_acts.long()
                new_logprob = pi.log_prob(act)
                entropy = pi.entropy()
                logratio = new_logprob - minibatch_logprobs
                ratio = logratio.exp()

                with th.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > clip_range).float().mean().item())

                if norm_adv:
                    minibatch_advs = (minibatch_advs - minibatch_advs.mean()) / (minibatch_advs.std() + 1e-8)
                
                # policy loss
                pg_loss1 = -minibatch_advs * ratio
                pg_loss2 = -minibatch_advs * th.clamp(ratio, 1-clip_range, 1+clip_range)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # value loss
                if clip_vloss:
                    val_loss_unclipped = (new_val - minibatch_rets)**2
                    val_clipped = minibatch_vals + th.clamp(new_val-minibatch_vals, -clip_range, clip_range) # todo: this clipping is different (not added with 1)
                    val_loss_clipped = (val_clipped - minibatch_rets)**2
                    val_loss_max = th.max(val_loss_unclipped, val_loss_clipped)
                    val_loss = 0.5 * val_loss_max.mean()
                else:
                    val_loss = 0.5 * ((new_val - minibatch_rets)**2).mean()
                
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


def learn_old(
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

    # todo: make larger buffer?
    # why should the buffer length be smaller than the total number of steps in an episode?

    obss = th.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    rews = th.zeros((num_steps, num_envs)).to(device)
    dones = th.zeros((num_steps, num_envs)).to(device)
    acts = th.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = th.zeros((num_steps, num_envs)).to(device)
    vals = th.zeros((num_steps, num_envs)).to(device)

    obs = th.tensor(envs.reset(), dtype=th.float).to(device)
    done = th.zeros(envs.num_envs).to(device)

    pbar = tqdm(total=tot_timesteps)

    timestep = 0

    for _batch in range(num_batches):
        # rollout steps
        for step in range(num_steps):
            timestep += num_envs

            with th.no_grad():
                pi, v = agent(obs)
                val = v.flatten()
                act = pi.sample()
                logprob = pi.log_prob(act)
            
            next_obs, rew, next_done, info = envs.step(act.cpu().numpy())
            rew = th.tensor(rew).to(device).view(-1)

            obss[step] = obs
            rews[step] = rew
            dones[step] = done
            acts[step] = act
            logprobs[step] = logprob
            vals[step] = val

            obs = th.tensor(next_obs, dtype=th.float).to(device)
            done = th.tensor(next_done, dtype=th.float).to(device)

            for i in info:
                if "episode" in i:
                    ep_r = i["episode"]["r"]
                    ep_l = i["episode"]["l"]
                    _ep_t = i["episode"]["t"]

                    writer.add_scalar("charts/episode_return", ep_r, timestep)
                    writer.add_scalar("charts/episode_length", ep_l, timestep)


        # bootstrap value
        with th.no_grad():
            _, next_val = agent(obs)
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
        batch_idx = np.arange(batch_size)
        batch_obss = obss.reshape((-1,) + envs.single_observation_space.shape)
        batch_acts = acts.reshape((-1,) + envs.single_action_space.shape)
        batch_logprobs = logprobs.reshape((-1,))
        batch_vals = vals.reshape((-1,))
        batch_advs = advs.reshape((-1,))
        batch_rets = rets.reshape((-1,))

        clip_fracs = []

        for _epoch in range(num_epochs):
            np.random.shuffle(batch_idx)

            for minibatch in range(num_minibatches):
                minibatch_idx = batch_idx[minibatch*minibatch_size:(minibatch+1)*minibatch_size]
                minibatch_obss = batch_obss[minibatch_idx]
                minibatch_acts = batch_acts[minibatch_idx]
                minibatch_logprobs = batch_logprobs[minibatch_idx]
                minibatch_vals = batch_vals[minibatch_idx]
                minibatch_advs = batch_advs[minibatch_idx]
                minibatch_rets = batch_rets[minibatch_idx]

                pi, v = agent(minibatch_obss)
                new_val = v.view(-1)
                act = minibatch_acts.long()
                new_logprob = pi.log_prob(act)
                entropy = pi.entropy()
                logratio = new_logprob - minibatch_logprobs
                ratio = logratio.exp()

                with th.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > clip_range).float().mean().item())

                if norm_adv:
                    minibatch_advs = (minibatch_advs - minibatch_advs.mean()) / (minibatch_advs.std() + 1e-8)
                
                # policy loss
                pg_loss1 = -minibatch_advs * ratio
                pg_loss2 = -minibatch_advs * th.clamp(ratio, 1-clip_range, 1+clip_range)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # value loss
                if clip_vloss:
                    val_loss_unclipped = (new_val - minibatch_rets)**2
                    val_clipped = minibatch_vals + th.clamp(new_val-minibatch_vals, -clip_range, clip_range) # todo: this clipping is different (not added with 1)
                    val_loss_clipped = (val_clipped - minibatch_rets)**2
                    val_loss_max = th.max(val_loss_unclipped, val_loss_clipped)
                    val_loss = 0.5 * val_loss_max.mean()
                else:
                    val_loss = 0.5 * ((new_val - minibatch_rets)**2).mean()
                
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
