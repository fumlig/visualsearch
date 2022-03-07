import pandas as pd
import json
import cv2 as cv
import ast

from os import path
from torch.utils.data import Dataset, DataLoader


DATA_PATH = path.join(path.dirname(__file__), "..", "data")


class AirbusDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = pd.read_csv(path.join(root_dir, "annotations.csv"), )

    def __len__(self):
        return len(self.data.image_id.unique())
    
    def __getitem__(self, idx):
        image_id = self.data.image_id.unique()[idx]
        image = cv.imread(path.join(self.root_dir, "images", image_id))
        targets = list(self.data[self.data["image_id"] == image_id].geometry.apply(ast.literal_eval).apply(lambda x: (x[0], x[2])))
        return image, targets