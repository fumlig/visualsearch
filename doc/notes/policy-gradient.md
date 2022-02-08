# Policy Gradient Methods for Reinforcement Learning with Function Approximation

- https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf

## Abstract

- Function approximation essential to RL
- Approximating value function and using it for policy proven intractable
- Alternative is to represent policy by its own function approximator
- Independent of the value function
- Update policy parameters with gradient ascent

## Introduction

- Value-function for actions previously dominant
- Policy selects action with highest value
- Oriented towards finding deterministic policies
- Optimal policy often stochastic
- An arbitrarily small change in estimated value can change action selection
- Obstacle for convergence assurances
- Rather than approximate value function, approximate stochastic policy directly
- Policy might be represented by neyral network with weights as policy parameters
- $\theta$ policy parameters, $\rho$ performance of policy (e.g. average reward per step)
- Policy gradient: update according to $\Delta \theta \approx \alpha \frac{\partial{\rho}}{\partial{\theta}}$
- Policy parameters can usually be assured to converge to a localy optimal policy in the performance measure.
- Unbiased estimate of gradient can be obtained from experiance using an approximate value function satisfying certain properties.
- REINFORCE also does this, but not with learned value function which means that it learns more slowly.
- Learning a value function and using it to reduce variance of gradient estimate seems to help rapid learning.
- May also prove convergence of both actor-critic and policy iteration architectures.

## Theorem

- Policy Gradient Theorem