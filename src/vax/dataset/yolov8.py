from vax.dataset import Dataset, DatasetConfig, Object, Label
from pathlib import Path
from vax.console import console
import yaml
SUBFOLDERS = ["train", "valid", "test"]

def convert_yolov8_det_dataset(yolov8_path: Path, output_path: Path):

    with open(yolov8_path / "data.yaml") as f:
        yolov8_config = yaml.safe_load(f)


    config = DatasetConfig(
        object_labels=yolov8_config["names"],
    )

    dataset = Dataset(output_path, config)

    subfolders = [folder for folder in SUBFOLDERS if (yolov8_path / folder).exists()]
    
    for subfolder in subfolders:
        images_path = yolov8_path / subfolder / "images"
        labels_path = yolov8_path / subfolder / "labels"

        for image_path in images_path.iterdir():
            label_path = labels_path / (image_path.stem + ".txt")
            with open(label_path) as f:
                label_text = f.read()

            detections = label_text.strip().split("\n")
            objects: list[Object] = []
            for detection in detections:
                cls, x_center, y_center, width, height = map(float, detection.split())
                # Convert from x_center, y_center, width, height to x_min, y_min, x_max, y_max
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                box = (x_min, y_min, x_max, y_max)
                objects.append((int(cls), box))

            label = Label(set=subfolder,objects=objects)
            
            dataset.write_item(image_path, label)