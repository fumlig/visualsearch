import gym
import numpy as np
import torch as th
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm
from ac import ActorCritic


def learn(
    envs,
    agent,
    device,
    writer=None,
    # hyperparameters
    learning_rate=5e-4,
    tot_timesteps=10e6,
    num_steps=256,
    num_minibatches=8,
    num_epochs=3,
    gamma=0.999,
    gae_lambda=0.95,
    norm_adv=True,
    clip_range=0.2,
    clip_vloss=False,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    seed=0
):
    num_envs = envs.num_envs
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches

    hparams = locals()


    envs.seed(seed)


    optimizer = th.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|" +
        "\n".join([f"|{param}|{value}|" for param, value in hparams.items()]) 
    )


    obss = th.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    rews = th.zeros((num_steps, num_envs)).to(device)
    dones = th.zeros((num_steps, num_envs)).to(device)
    acts = th.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = th.zeros((num_steps, num_envs)).to(device)
    vals = th.zeros((num_steps, num_envs)).to(device)

    obs = th.tensor(envs.reset(), dtype=th.float).to(device)
    done = th.zeros(envs.num_envs).to(device)

    for timestep in tqdm(range(0, tot_timesteps, batch_size)):
        
        # rollout
        for step in range(num_steps):
            
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

            for env_info in info:
                if "episode" in env_info:
                    episode_r = env_info["episode"]["r"]
                    episode_l = env_info["episode"]["l"]

                    writer.add_scalar("charts/episode_reward", episode_r, timestep)
                    writer.add_scalar("charts/episode_length", episode_l, timestep)
        
        # bootstrap
        with th.no_grad():
            _, next_val = agent(next_obs)
            advs = th.zeros_like(rews)
            last_gae_lambda = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_val = next_val.reshape(1, -1)
                    next_nonterminal = 1.0 - next_done
                else:
                    next_val = vals[t+1]
                    next_nonterminal = 1.0 - dones[t+1]

                delta = rews[t] + gamma*next_val*next_nonterminal - vals[t]
                last_gae_lambda = delta + gamma*gae_lambda*next_nonterminal*last_gae_lambda
                advs[t] = last_gae_lambda

            rets = advs + vals
        
        # train
        batch_obss = obss.reshape((-1) + envs.single_observation_space.shape)
        batch_rews = rews.reshape((-1))
        batch_dones = dones.reshape((-1))
        batch_acts = acts.reshape((-1,) + envs.single_action_space.shape)
        batch_logprobs = logprobs.reshape((-1))
        batch_vals = vals.reshape((-1))
        batch_advs = advs.reshape((-1))
        batch_rets = rets.reshape((-1))

        clip_fracs = []

        for epoch in range(num_epochs):
            batch_idx = np.random.shuffle(np.arange(batch_size))
            
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
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > clip_range).float().mean().item())


            
                if norm_adv:
                    minibatch_advs = (minibatch_advs - minibatch_advs.mean()) / (minibatch_advs.std() + 1e-8)
                
                pg_loss1 = -minibatch_advs*ratio
                pg_loss2 = -minibatch_advs * th.clamp(ratio, 1-clip_range, 1+clip_range)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

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
        pred_vals = batch_vals.cpu().numpy()
        true_vals = batch_rets.cpu().numpy()
        
        writer.add_scalar("losses/value_loss", val_loss.item(), timestep)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), timestep)
        writer.add_scalar("losses/entropy_loss", ent_loss.item(), timestep)

        explained_variance = np.var()

if __name__ == "__main__":

    import gym_search

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    envs = gym.vector.make(
        "SearchSparse-v0",
        64,
        asynchronous=True,
        wrappers=[gym.wrappers.FlattenObservation, gym.wrappers.RecordEpisodeStatistics, gym.wrappers.NormalizeReward]
    )
    agent = ActorCritic(envs)

    writer = SummaryWriter("test/test")

    learn(envs, agent, device, writer)

    envs.close()
    writer.close()