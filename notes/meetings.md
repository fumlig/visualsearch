# 2022-02-09

Meeting with Sourabh, some general questions:

- Explain the current plan
- Solve an object search problem
- A camera can look around in an environment
- Receives image observations
- It is expected to locate target objects placed in the environment
- Can be referred to as visual search or object search
- The environment is partially visible

- The reason for using RL for this task is that it can potentially be more dynamic
- Say for example that the environment changes or the appearance/placement of target changes
- A solution using domain knowledge would need to be reconfigured each time
- RL could avoid this

- We had an environment for this but it is quite resource demanding
- A previous master's student used it for something similar and one training run took multiple days
- This is something I want to avoid
- Furthermore, the results that time were not great, and the cause of these issues were difficult to pinpoint

- My plan is to instead set up a simpler environment in which the desired agent characteristics can still be tested
- This environment can be faster to experiment with and training times should be lower
- Targets are more likely to be placed in some locations than others (for example, less likely that targets are in the sky and more likely that they are hiding behind something)
- The environment is partially observable and the view has to be moved around
- There is some incentive to zoom in on the target
- Multiple targets may be placed around the agent and it should learn to find all of them and indicate when it is done
- Some additional things that could be looked at include:
- Targets appearing over time: the agent has to learn some patrolling behaviour and regularly cover the whole environment

- I have formulated some research questions:

- How can an RL agent like this be implemented? (Algorithm, reward signal, observation space, action space, architecture, feature extraction with RNN, CNN etc. Could potentially compare different architectures.)
- How does an RL agent compare to a baseline agent (rule-based, random, human-controlled, etc.)

- Could the environment construction be made into a research question (how can an environment that tests these requirements be implemented? a little vague...)


- It is unclear how far I can get, how many of the problems I can solve
- Is it reasonable to leave that unspecified and set a deadline for myself, after which I 


Contributions:
- Environment
- Methodology
- Analysis of methodology

- Focus on method, not results

Problem:
- Environment and methodology could be too much
- Make environment the simpler task, allocate less time (1 month)
- Focus of work is the methodology: reward function