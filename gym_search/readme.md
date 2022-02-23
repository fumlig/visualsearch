# search

- We should fix training set size, and do tests with different sizes
- It is not feasible to have an infinite number of environments
- If we restrict training to a finite set, this solves the question about knowing the distribution;
- A fixed number of environments could be collected feasibly without having the distribution being known.
- At test time, zero-shot generalization is averaged


- fixed seed pool for training
- think about max timesteps - could it be lowered (for example to the number of tiles)?