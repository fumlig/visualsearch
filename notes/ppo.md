# Proximal Policy Optimization

- https://openai.com/blog/openai-baselines-ppo/
- https://github.com/openai/baselines
- https://arxiv.org/abs/1707.06347

Policy gradient methods are challenging because they are sensitive to choices of step size. Compared to supervised learning, RL is less predictable and difficult to debug. PPO tries to address these issues.

The loss over the policy parameters is defined as the empirical expectation over timesteps of the estimated advantage at each time step.

- Sort of a mix between on- and off-policy, since it uses old samples?


Behaviour:

- Entropy coefficient seems to matter
- Higher means more exploration
- Too high can cause agent to forget?
- With good observations, actions and rewards should work


- Deals with vector environments: i.e. is build for usage with vector environments.

Interesting thought regarding unlearning/instability:

- https://www.reddit.com/r/reinforcementlearning/comments/d3wym2/catastrophic_unlearning_in_ppo_a_plausible_cause/


- How to give more weight to replay buffer as time goes on?
  https://arxiv.org/pdf/2112.03798.pdf

- Decide on one or two algorithms is probably best.