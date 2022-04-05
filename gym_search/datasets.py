import pandas as pd
import geopandas as geopd
import json
import cv2 as cv
import ast
import os

from torch.utils.data import Dataset

"""
if os.path.exists("data/airbus-aircraft"):
    gym.register(
        id="SearchAirbusAircraft-v0",
        entry_point=SearchEnv,
        kwargs=dict(
            generator=DatasetGenerator(AirbusAircraftDataset("data/airbus-aircraft")),
            view_shape=(128, 128),
            step_size=128
        )
    )

if os.path.exists("data/airbus-oil"):
    gym.register(
        id="SearchAirbusOil-v0",
        entry_point=SearchEnv,
        kwargs=dict(
            generator=DatasetGenerator(AirbusOilDataset("data/airbus-oil")),
            view_shape=(128, 128),
            step_size=128
        )
    )
"""


def generate_dataset(root, generator, num_train, num_test, seed=0):    
    generator.seed(seed)

    train_items = []
    train_dir = os.path.join(root, "train")
    os.makedirs(train_dir, exist_ok=True)

    for i in range(num_train):
        img, tgts = generator.sample()
        img_path =  os.path.join(train_dir, f"{i}.png")
        cv.imwrite(img_path, img)
        train_items.append({
            "id": i,
            "targets": [(tgt.y, tgt.x, tgt.h, tgt.w) for tgt in tgts]
        })


class XViewDataset(Dataset):
    def __init__(self, root="data/view"):
        self.root = root
        self.data = geopd.read_file("data/xview/train_labels/xView_train.geojson")

    def __len__(self):
        return len(self.data["image_id"].unique())
    
    def __getitem__(self, idx):
        image_id = self.data.image_id.unique()[idx]
        image = cv.imread(os.path.join(self.root, "train_images", "train_images", image_id))
        coords = list(self.data[self.data["image_id"] == image_id]["bounds_imcoords"].apply(lambda x: tuple(map(int, x.split(",")))))

        targets = []
        for x0, y0, x1, y1 in coords:
            x, y = x0, y0
            w, h = x1 - x0, y1 - y0
            targets.append(((y, x), (h, w)))

        return image, targets

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
    