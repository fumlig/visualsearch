import torch as th
from torch.profiler import profile, record_function, ProfilerActivity


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        #model(input)
        pass