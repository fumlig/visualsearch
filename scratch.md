meeting with sourabh:
explain what's happening

1. started with searching for targets with camera
2. now we have ended up with the following task
  - there is an observer that observes part of some scene
  - it can transform its view with a set of actions
  - there is a number of targets in the scene
  - the goal of the agent is to bring each target into view with a minimal number of actions
  - the agent has knowledge of the boundaries of what it can currently observe
    - it can therefore keep track of what it has seen
  - we want a learning agent for this because
    - it is interesting to see whether it is feasible
    - might be difficult to "manually" create an agent for every type of environment
    - optimal search patterns may be complex, subtle, etc.
3. the main contribution of my thesis will be a method to solve this problem
  - it is therefore quite "practical" thesis, not very theoretical
  - reinforcement learning applied to a (somewhat) real problem
  - applications in for example search-and-rescue, helping robots, etc.
4. I have gotten pretty far with the implementation up with something that I think could be good
  - a reinforcement learning agent
  - two kinds of observations
  - one RGB image of the currently visible part of the scene
  - one feature map (stacked binary masks) that contain information about where the agent is, what it has explored, etc.
  - neural network architecture
  - have tried this on the simple environment at it looks promising
5. what I am checking now is how far the method scales, how difficult problems it can be applied to
  - plan is to use three environments of varying difficulty
  - show these environments (the first and second seem to be solvable well)
  - would ideally like to have some 3d environment as the final one
  - another option is to use images from some dataset
6. research questions
  - how can an agent that learns to search be implemented
  - how does the learning method compare to certain non-learning methods (random walk, exhaustive search, human searcher. possible some other method but haven't found a suitable one yet)
  - how well does the method generalize to unseen environments
    - many works in deep reinforcement learning either use one environment (for example atari game), or a small set
    - this means that one can not know if the agent has actually learned a general policy or not
    - partly due to lack of god environments to test generalization
    - there are a few recent works that study generalization using procedurally generated environments
    - I thought that this would be suitable for this problem,
    - because the idea is to train the agent with some limited set of samples and then use it in new environments 
    - need a controllable environment for this one, have a simple terrain test
    - plan is to train the agent for a fixed number of time steps
    - vary the number of available samples during training
    - test on full distribution of environments
    - see how performance improves with training set size
    - of course dependent on problem difficulty (the environment itself) but seeing how it changes should still be interesting

  - my current thinking is to use only one training algorithm (PPO) and one method (architecture, reward function etc)
  - try to make this one as good as possible and then test it in depth
  - could potentially compare multiple algorithms but this would take time and not sure that it is that interesting (PPO is SOTA model-free)

7. half-time presentation
  - I am in week 7 now
  - will reach out to jose about having meeting in week 11
8. next steps for me:
  - collect measurements for first environment to have something concrete
  - spend more time on report to finish introduction and theory
  - collect results on generalization
  - determine good third environment/direction
    - 3D, or
    - real photos
    - something else
  - test method on it


send repo link to sourabh
there are many tangents I could go on in report
many similar tasks that are related but not exactly the same