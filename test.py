import torch
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(f"logs/test")

print("adding video")
writer.add_video(
    "videos/test",
    torch.rand((1, 16, 3, 10, 10)),
    global_step=0,
    fps=4
)
print("adding video")
