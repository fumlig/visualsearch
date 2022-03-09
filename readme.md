# Master's Thesis

Learning to Search in High-Dimensional Signals

- rewrite to n-dimensional signals
- environment should by default maybe just return signal, use wrappers for position and visited and so on
- spend some time on report and reach out to supervisor and examiner with update
- figure out why policy search is useful
  - "guided policy search is a method for performing policy search in continuous state and action spaces under possibly unknown dynamics."
- think through rewards and actions *thoroughly* and note down thoughts
- add support for transforms from torchvision, scikit-image has:
  - hog: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
  - sift: https://scikit-image.org/docs/0.19.x/auto_examples/features_detection/plot_sift.html
- the datasets (usb and airbus are very small, maybe take some precautions to increase their size?)
  - torchvision has some data augmentation techniques: https://pytorch.org/vision/stable/transforms.html