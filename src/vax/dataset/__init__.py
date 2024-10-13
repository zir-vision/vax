from pathlib import Path
from pydantic import BaseModel, Field
import yaml
from typing import Generator
import shutil
from abc import ABC, abstractmethod
import jax.numpy as jnp

Box = tuple[float, float, float, float]
Object = tuple[int, Box]


class Label(BaseModel):
    set: str
    objects: list[Object] | None = None
    cls: int | None = None


class DatasetConfig(BaseModel):
    object_labels: list[str] | None = None
    classification_labels: list[str] | None = None


class Dataset:
    path: Path
    config: DatasetConfig

    def __init__(self, path: str | Path, config: DatasetConfig):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.config = config

        (self.path / "images").mkdir(exist_ok=True)
        (self.path / "labels").mkdir(exist_ok=True)

    def save_config(self, config: DatasetConfig):
        with open(self.path / "config.yaml", "w") as f:
            yaml.safe_dump(config.model_dump(), f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        with open(path / "config.yaml") as f:
            config = yaml.safe_load(f)
        config = DatasetConfig.model_validate(config)
        return cls(path, config)

    def __iter__(self) -> Generator[tuple[Path, Label], None, None]:
        for image_path in (self.path / "images").iterdir():
            label_path = self.path / "labels" / (image_path.stem + ".txt")
            with open(label_path) as f:
                label = yaml.safe_load(f)
            label = Label.model_validate(label)
            yield image_path, label

    def write_item(self, image_path: Path, label: Label):
        label_path = self.path / "labels" / (image_path.stem + ".txt")
        with open(label_path, "w") as f:
            yaml.safe_dump(label.model_dump(), f)
        try:
            shutil.copy(image_path, self.path / "images" / image_path.name)
        except shutil.SameFileError:
            pass


class Decoder(ABC):
    @abstractmethod
    def __init__(self, ds: Dataset): ...

    @abstractmethod
    def __call__(self, image_path: Path, label: Label) -> dict: ...


class ClassificationDecoder(Decoder):
    def __init__(self, ds: Dataset):
        ...

    def __call__(self, image_path: Path, label: Label) -> dict:
        import cv2
        return {
            "image": jnp.array(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)),
            "label": label.cls
        }



class EagerDataloader(Dataset):
    def __init__(self, ds: Dataset, decoder: Decoder):
        self.ds = ds
        self.decoder = decoder
        self.sets = {}

        items = []

        for e in ds:
            self.sets.setdefault(e[1].set, []).append(e)
            items.append(e)

    def length(self, set: str):
        return len(self.sets[set])

    def load_item(self, set: str, index: int):
        return self.decoder(*self.sets[set][index])
    
    def iterate_set(self, set: str) -> Generator:
        for item in self.sets[set]:
            yield self.decoder(*item)

    def batch_set(self, set: str, batch_size: int) -> Generator[list[tuple[Path, Label]], None, None]:
        for i in range(0, len(self.sets[set]), batch_size):
            if i + batch_size > len(self.sets[set]):
                return
            batch = [self.decoder(*e) for e in self.sets[set][i : i + batch_size]]
            yield {
                "image": jnp.stack([e["image"] for e in batch]),
                "label": jnp.array([e["label"] for e in batch])
            }