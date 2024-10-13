"""
Parses the Fashion MNIST dataset.
Input is a csv file for each of the train and test (valid) sets.
For each row in the csv file, the first column is the label and the rest are pixel values.
Each image is 28x28 pixels.
"""

from vax.dataset import Dataset, DatasetConfig, Label
from pathlib import Path
import csv
import numpy as np
import cv2

def convert_fashion_mnist_dataset(train_path: Path, test_path: Path, output_path: Path):
    config = DatasetConfig(
        classification_labels=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    )

    dataset = Dataset(output_path, config)
    dataset.save_config(config)
    for path, set in [(train_path, "train"), (test_path, "valid")]:
        with open(path) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                label = int(row[0])
                image = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
                image_path = dataset.path / "images" / f"{i}.png"
                cv2.imwrite(str(image_path), image)
                label = Label(set=set, cls=label)
                dataset.write_item(image_path, label)