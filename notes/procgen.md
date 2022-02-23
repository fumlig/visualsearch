# ProgGen Benchmark

- Generalization and sample efficiency increase with model size
- 25M vs. 200M timesteps for easy vs hard
- Many more things are randomized, including appearance of objects.
- We would like to do this, but maybe start with not doing it...
- Could easily randomize appearance of objects


- Always Proximal Policy Optimization
- Sample efficiency evaluated by training and testing agents on full distribution of levels
- Generalization evaluated on finite training set (500) and test on full distribution of levels
- Hyperparameter tuning requirements minimized
- Framestacking minimally affects performance
- Use convolutional architecture in IMPALA

- Smaller architectures struggle to train when faced with high diversity
- Agents strongly overfit to small training sets
- Agents need access to as many as 10,000 levels to close generalization gap
- Past a certain threshold, training performance improves as the training set grows.
- Counter to trends in supervised learning where training performance can decrease with size of training set
- Attributed to implicit curriculum provided by distribution of levels.

- They sample a new level at the start of every episode
- Ablation studies with deterministic sequence:
- Start at level 1, resets if agent fails and episode terminates, goes to next level if successful
- Rarely got past level 20 for each episode
- Testing with non-deterministic shows that progress is made over training set but it is not meaningful: they learn almost nothing about underlying level distribution and fail on test levels.
- Emphasize importance of both training and testing with diverse environment distributions.

- They "do not restrict the training time, but in practice still train for 200M timesteps"
- What does this mean - are the 200M timesteps implicit from the level count?

- Pretty detailed model size tests
- 4 variants: 3x IMPALA and 1x Nature CNN
- Only vary learning rate between architectures
- Small nature-CNN almost fails to train