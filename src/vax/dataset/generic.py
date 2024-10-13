from vax.dataset import Dataset, DatasetConfig, Object, Label
from pathlib import Path
from vax.console import console
import yaml
SUBFOLDERS = ["train", "valid", "test"]

def convert_generic_classification_dataset(input_path: Path, output_path: Path):
    subfolders = [folder for folder in SUBFOLDERS if (input_path / folder).exists()]

    labels = set()

    for subfolder in subfolders:
        for label_path in (input_path / subfolder).iterdir():
            labels.add(label_path.name)

    config = DatasetConfig(
        classification_labels=list(labels)
    )

    print(config)

    dataset = Dataset(output_path, config)

    for subfolder in SUBFOLDERS:
        for label_path in (input_path / subfolder).iterdir():
            for image_path in label_path.iterdir():
                label = Label(set=subfolder, cls=config.classification_labels.index(label_path.name))
                dataset.write_item(image_path, label)
            