- Stable baselines 3 does the following to extract features from dictionary observations (`stable_baselines3.common.torch_layers.CombinedExtractor`): 

```py
"""
Combined feature extractor for Dict observation spaces.
Builds a feature extractor for each key of the space. Input from each space
is fed through a separate submodule (CNN or MLP, depending on input shape),
the output features are concatenated and fed through additional MLP network ("combined").
"""

import numpy as np
```

- Have seen papers include time step in observation, why is this done? Some discussion here: https://www.reddit.com/r/reinforcementlearning/comments/f6hw1v/include_step_number_in_state_variables/
- Can including time turn every problem into an MDP?
- https://arxiv.org/abs/1712.00378