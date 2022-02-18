- Visual search for multiple targets
- It is also target search in unknown environment...
- Searching problem explored in robotics and computer vision for stationary environments
- Dynamic environments put additional requirements
- One motivation could be that we want to deploy the same algorithm in similar but different environments

- Environment appearance correlated with probability of target
- Actions translate the agent horizontally or vertically: camera movements are relative, never absolute, and correspond to some cost (time, effort, etc.)
- Instance of [TSP](tsp.md).
- I want to go towards the theoretical direction. How much connection to retain to the original visual search problem?
- Define the problem mathematically? Could be fun. Would likely need some help... And more decisions.


- Mapping seems redundant, the environment always has the same size, and there are no movement restrictions.
- With appearing targets, it becomes useful to remember where you have been recently. Are recurrent neural networks sufficient for this?