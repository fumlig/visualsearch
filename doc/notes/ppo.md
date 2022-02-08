# Proximal Policy Optimization

- https://openai.com/blog/openai-baselines-ppo/
- https://github.com/openai/baselines
- https://arxiv.org/abs/1707.06347

Policy gradient methods are challenging because they are sensitive to choices of step size. Compared to supervised learning, RL is less predictable and difficult to debug. PPO tries to address these issues.

The loss over the policy parameters is defined as the empirical expectation over timesteps of the estimated advantage at each time step 