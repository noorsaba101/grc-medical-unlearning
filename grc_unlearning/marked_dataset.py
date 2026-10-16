"""In-memory marker views of the unmodified MedCoSS ChestXR_Dataset."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from . import CLASS_LABELS, DEFAULT_MARKER_MARGIN, DEFAULT_MARKER_SIZE, validate_marker_options
from .provenance import load_manifest
from .synthetic_marker import apply_marker


def create_base_dataset(dataset_root, split="train", crop_size=(224, 224)):
    """Reuse the original class, including its sample order, split, and transforms."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    from medcoss.Downstream.Dim_2.Chest_XR.dataloader import ChestXR_Dataset

    return ChestXR_Dataset(str(dataset_root), split=split, crop_size=crop_size)


def load_chestxr_image(path):
    """Match the original OpenCV grayscale -> PIL RGB decoding exactly."""
    import cv2

    grayscale = cv2.imread(str(path), 0)
    if grayscale is None:
        raise OSError("could not decode ChestXR image: {}".format(path))
    return Image.fromarray(grayscale).convert("RGB")


class MarkedChestXRDataset(Dataset):
    """Wrap an existing ChestXR_Dataset without changing its transforms or labels.

    Modes are 'manifest', 'force_on', and 'force_off'. Manifest mode uses each
    selected record's marker settings. Force-on marks every sample with the
    constructor's settings. Force-off delegates directly to the original dataset.
    Missing train/val provenance is an error; missing test records mean clean.
    Markers are drawn before the original transform, so training augmentations
    can blur or flip them just as they would any other image content.
    """

    def __init__(
        self,
        base_dataset,
        manifest_path=None,
        marker_mode="manifest",
        marker_type="square",
        marker_size=DEFAULT_MARKER_SIZE,
        marker_margin=DEFAULT_MARKER_MARGIN,
    ):
        if marker_mode not in ("manifest", "force_on", "force_off"):
            raise ValueError("marker_mode must be manifest, force_on, or force_off")
        validate_marker_options(marker_type, marker_size, marker_margin)
        if marker_mode == "manifest" and manifest_path is None:
            raise ValueError("manifest mode requires manifest_path")
        if base_dataset.classes != CLASS_LABELS:
            raise ValueError("expected covid=0, normal=1, pneumonia=2")
        if len(base_dataset.train_samples) != len(base_dataset.train_labels):
            raise ValueError("base dataset sample and label lengths differ")

        self.base_dataset = base_dataset
        self.marker_mode = marker_mode
        self.marker_type = marker_type
        self.marker_size = marker_size
        self.marker_margin = marker_margin
        records = load_manifest(manifest_path) if manifest_path is not None else []
        self.records_by_path = {record["relative_path"]: record for record in records}
        root = Path(base_dataset.data_path).resolve()
        self.relative_paths = [
            Path(path).resolve().relative_to(root).as_posix()
            for path in base_dataset.train_samples
        ]
        for index, relative_path in enumerate(self.relative_paths):
            record = self.records_by_path.get(relative_path)
            if record is None:
                if manifest_path is not None and relative_path.startswith("train/"):
                    raise ValueError("missing training/validation provenance: {}".format(relative_path))
            elif record["label"] != base_dataset.train_labels[index]:
                raise ValueError("manifest label differs from dataset: {}".format(relative_path))

    @classmethod
    def from_root(cls, dataset_root, manifest_path=None, split="train", crop_size=(224, 224), **kwargs):
        base_dataset = create_base_dataset(dataset_root, split=split, crop_size=crop_size)
        return cls(base_dataset, manifest_path=manifest_path, **kwargs)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        record = self.records_by_path.get(self.relative_paths[index])
        marked = self.marker_mode == "force_on" or (
            self.marker_mode == "manifest" and record is not None and record["marker_present"]
        )
        if not marked:
            return self.base_dataset[index]

        marker_type, size, margin = self.marker_type, self.marker_size, self.marker_margin
        if self.marker_mode == "manifest":
            marker_type = record["marker_type"]
            size = record.get("marker_size", size)
            margin = record.get("marker_margin", margin)
        image = load_chestxr_image(self.base_dataset.train_samples[index])
        image = apply_marker(image, marker_type=marker_type, size=size, margin=margin)
        image = self.base_dataset.transform(image)
        label = torch.tensor([float(self.base_dataset.train_labels[index])])
        return image, label


class PairedChestXRDataset(Dataset):
    """Return (clean_tensor, marked_tensor, label) for each original test image.

    Both views share one base test dataset, sample order, and deterministic test
    preprocessing. All test classes receive a marked counterfactual regardless
    of training provenance. No test labels or files are changed.
    """

    def __init__(
        self,
        dataset_root,
        crop_size=(224, 224),
        marker_type="square",
        marker_size=DEFAULT_MARKER_SIZE,
        marker_margin=DEFAULT_MARKER_MARGIN,
    ):
        base = create_base_dataset(dataset_root, split="test", crop_size=crop_size)
        self.clean = MarkedChestXRDataset(base, marker_mode="force_off")
        self.marked = MarkedChestXRDataset(
            base, marker_mode="force_on", marker_type=marker_type,
            marker_size=marker_size, marker_margin=marker_margin,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean_image, label = self.clean[index]
        marked_image, _ = self.marked[index]
        return clean_image, marked_image, label
