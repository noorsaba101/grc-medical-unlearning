"""In-memory marker views of the unmodified MedCoSS ChestXR_Dataset."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import CLASS_LABELS, DEFAULT_MARKER_MARGIN, DEFAULT_MARKER_SIZE, MARKER_PLACEMENTS, validate_marker_options
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
    Default 'native' placement preserves the original pre-transform marker.
    'post_augmentation' inserts it after all original PIL operations and directly
    before the final ToTensor. Explicit marker_overrides change appearance only;
    manifest selection and labels are never overridden.
    """

    def __init__(
        self,
        base_dataset,
        manifest_path=None,
        marker_mode="manifest",
        marker_type="square",
        marker_size=DEFAULT_MARKER_SIZE,
        marker_margin=DEFAULT_MARKER_MARGIN,
        marker_placement="native",
        marker_overrides=None,
    ):
        if marker_mode not in ("manifest", "force_on", "force_off"):
            raise ValueError("marker_mode must be manifest, force_on, or force_off")
        validate_marker_options(marker_type, marker_size, marker_margin)
        if marker_placement not in MARKER_PLACEMENTS:
            raise ValueError("unsupported marker_placement: {}".format(marker_placement))
        marker_overrides = dict(marker_overrides or {})
        if set(marker_overrides) - {"marker_type", "marker_size", "marker_margin"}:
            raise ValueError("marker_overrides may change only marker appearance, not sample selection")
        effective = dict(marker_type=marker_type, marker_size=marker_size, marker_margin=marker_margin)
        effective.update(marker_overrides)
        validate_marker_options(effective["marker_type"], effective["marker_size"], effective["marker_margin"])
        if marker_placement == "post_augmentation" and marker_mode != "force_off":
            self._validate_post_augmentation_transform(base_dataset.transform)
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
        self.marker_placement = marker_placement
        self.marker_overrides = marker_overrides
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

    @staticmethod
    def _validate_post_augmentation_transform(transform):
        if (not isinstance(transform, transforms.Compose) or not transform.transforms
                or not isinstance(transform.transforms[-1], transforms.ToTensor)
                or sum(isinstance(op, transforms.ToTensor) for op in transform.transforms) != 1):
            raise ValueError("post_augmentation requires the original Compose ending with exactly one ToTensor")

    def marker_parameters(self, record=None):
        """Resolve rendering settings independently of immutable membership."""
        values = {"marker_type": self.marker_type, "marker_size": self.marker_size,
                  "marker_margin": self.marker_margin}
        if self.marker_mode == "manifest" and record is not None:
            values.update(marker_type=record["marker_type"], marker_size=record.get("marker_size", self.marker_size),
                          marker_margin=record.get("marker_margin", self.marker_margin))
        values.update(self.marker_overrides)
        return values["marker_type"], values["marker_size"], values["marker_margin"]

    def __getitem__(self, index):
        record = self.records_by_path.get(self.relative_paths[index])
        marked = self.marker_mode == "force_on" or (
            self.marker_mode == "manifest" and record is not None and record["marker_present"]
        )
        if not marked:
            return self.base_dataset[index]

        marker_type, size, margin = self.marker_parameters(record)
        image = load_chestxr_image(self.base_dataset.train_samples[index])
        if self.marker_placement == "post_augmentation":
            transform = self.base_dataset.transform
            self._validate_post_augmentation_transform(transform)
            for operation in transform.transforms[:-1]:
                image = operation(image)
            if not isinstance(image, Image.Image) or image.mode != "RGB":
                raise ValueError("expected an augmented PIL RGB image immediately before ToTensor")
            image = apply_marker(image, marker_type=marker_type, size=size, margin=margin)
            image = transform.transforms[-1](image)
        else:
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
        marker_placement="native",
    ):
        base = create_base_dataset(dataset_root, split="test", crop_size=crop_size)
        self.clean = MarkedChestXRDataset(base, marker_mode="force_off")
        self.marked = MarkedChestXRDataset(
            base, marker_mode="force_on", marker_type=marker_type,
            marker_size=marker_size, marker_margin=marker_margin, marker_placement=marker_placement,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean_image, label = self.clean[index]
        marked_image, _ = self.marked[index]
        return clean_image, marked_image, label


class PairedTrainingChestXRDataset(Dataset):
    """One augmented radiograph, then clean/marked views for concept unlearning.

    Every training sample, in every class, receives a counterfactual marked view.
    This does not change Stage 1/2 provenance membership. Original PIL operations
    execute ONCE, so the paired tensors differ only inside the forbidden marker.
    """

    def __init__(self, base_dataset, marker_type="square", marker_size=32, marker_margin=8,
                 marker_placement="post_augmentation"):
        if marker_placement != "post_augmentation":
            raise ValueError("paired training requires post_augmentation placement")
        validate_marker_options(marker_type, marker_size, marker_margin)
        MarkedChestXRDataset._validate_post_augmentation_transform(base_dataset.transform)
        if base_dataset.classes != CLASS_LABELS:
            raise ValueError("expected covid=0, normal=1, pneumonia=2")
        if len(base_dataset.train_samples) != len(base_dataset.train_labels):
            raise ValueError("base dataset sample and label lengths differ")
        self.base_dataset = base_dataset
        self.marker_type, self.marker_size, self.marker_margin = marker_type, marker_size, marker_margin
        self.marker_placement = marker_placement
        root = Path(base_dataset.data_path).resolve()
        self.relative_paths = [Path(path).resolve().relative_to(root).as_posix()
                               for path in base_dataset.train_samples]
        if any(not path.startswith("train/") for path in self.relative_paths):
            raise ValueError("paired training must not contain validation-folder test images")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        transform = self.base_dataset.transform
        MarkedChestXRDataset._validate_post_augmentation_transform(transform)
        image = load_chestxr_image(self.base_dataset.train_samples[index])
        for operation in transform.transforms[:-1]:
            image = operation(image)
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("expected an augmented PIL RGB image immediately before ToTensor")
        marked = apply_marker(image, marker_type=self.marker_type, size=self.marker_size, margin=self.marker_margin)
        label = torch.tensor([float(self.base_dataset.train_labels[index])])
        return transform.transforms[-1](image), transform.transforms[-1](marked), label
