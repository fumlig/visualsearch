import pandas as pd
import json
import cv2 as cv
import ast
import os

from torch.utils.data import Dataset


class AirbusAircraftDataset(Dataset):
    # https://www.kaggle.com/airbusgeo/airbus-aircrafts-sample-dataset/download

    def __init__(self, root="data/airbus-aircraft"):
        self.root = root
        self.data = pd.read_csv(os.path.join(root, "annotations.csv"))

    def __len__(self):
        return len(self.data.image_id.unique())
    
    def __getitem__(self, idx):
        image_id = self.data.image_id.unique()[idx]
        image = cv.imread(os.path.join(self.root, "images", image_id))
        geometries = list(self.data[self.data["image_id"] == image_id].geometry.apply(ast.literal_eval))

        targets = []
        for geometry in geometries:
            top_left = geometry[0]
            bottom_right = geometry[2]
            x, y = top_left
            w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
            targets.append(((y, x), (h, w)))

        return image, targets


class AirbusOilDataset(Dataset):
    # https://www.kaggle.com/airbusgeo/airbus-oil-storage-detection-dataset

    def __init__(self, root="data/airbus-oil"):
        self.root = root
        self.data = pd.read_csv(os.path.join(root, "annotations.csv"))

    def __len__(self):
        return len(self.data.image_id.unique())
    
    def __getitem__(self, idx):
        image_id = self.data.image_id.unique()[idx]
        image = cv.imread(os.path.join(self.root, "images", f"{image_id}.jpg"))
        bounds = list(self.data[self.data["image_id"] == image_id].bounds.apply(ast.literal_eval))

        targets = []
        for x0, y0, x1, y1 in bounds:
            x, y = x0, y0
            w, h = x1-x0, y1-y0
            targets.append(((y, x), (h, w)))

        return image, targets
    