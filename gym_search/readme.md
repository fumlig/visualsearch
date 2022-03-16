# search

- We should fix training set size, and do tests with different sizes
- It is not feasible to have an infinite number of environments
- If we restrict training to a finite set, this solves the question about knowing the distribution;
- A fixed number of environments could be collected feasibly without having the distribution being known.
- At test time, zero-shot generalization is averaged

- one bird's eye view
- one classic visual search
- one horizon

- bring back zoom?
- zoom simply oversamples pixels
- moving camera when zoomed in moves it less (= one view width)
- costly to move when zoomed in


- what if we don't indicate that a target has been found? actually makes more sense

- have one realistic dataset?
- https://github.com/michaelthoreau/SearchAndRescueNet
- https://www.kaggle.com/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects
  - very similar images



- https://www.nuscenes.org/nuimages
- https://www.cityscapes-dataset.com/
  - available via torchvision
- https://cocodataset.org/#captions-2015
- https://www.yf.io/p/lsun
- http://places2.csail.mit.edu/index.html
- map data? 


- https://arxiv.org/pdf/2107.12469.pdf

- add bounding box class
- create dataset from object detection dataset
- fun extra: search for words in text?
- give low-res glimpse of the environment beforehand, where targets are not visible?

- oddity search

1. basic search
2. terrain search
3. complex search
  - try to make attention crucial
  - some procedurally generated texture?
  - we can
  - something with roads?
  - or skip attention, difficult to make time for

4. some kind of room


## Inspiration

https://github.com/s-macke/VoxelSpace