import os
import numpy as np
import torch
import torch.utils.data as data
import json
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PATH_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def is_pathology_image_file(filename):
    return os.path.splitext(filename.lower())[1] in PATH_IMAGE_EXTENSIONS


def load_pathology_image_pil_rgb(img_path):
    """Load pathology patch as RGB PIL; cv2 first, PIL fallback for TIF and failures."""
    arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if arr is not None and arr.size > 0:
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            return Image.fromarray(arr)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(arr)
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(
            f"Failed to read pathology image {img_path!r} (cv2 and PIL failed): {e}"
        ) from e


def collect_pathology_image_paths(data_path, list_filename="pretrain_data_list.json"):
    list_path = os.path.join(data_path, list_filename)
    if os.path.exists(list_path):
        with open(list_path, "r") as f:
            return json.load(f)["path"]
    paths = []
    for root, dirs, files in tqdm(os.walk(data_path), desc="scan pathology"):
        for fname in files:
            if fname == list_filename:
                continue
            if is_pathology_image_file(fname):
                paths.append(os.path.join(root, fname))
    data_list = {"path": paths}
    with open(list_path, "w") as f:
        json.dump(data_list, f, sort_keys=True, indent=4)
    return paths


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

class TCGA_Image_Dataset(data.Dataset):
    def __init__(self, data_path, imsize=(224, 224), is_sort=False):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.image_path = collect_pathology_image_paths(data_path)

        self.tr_transforms2D = get_train_transform2D(imsize)

        if is_sort:
            self.img_ids = sorted(self.image_path)
            print("sorted dataset")

        print("image sample number: ", len(self.image_path))

    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, index):
        img_path = self.image_path[index]
        image2D = load_pathology_image_pil_rgb(img_path)
        image2D_trans = self.tr_transforms2D(image2D)


        return image2D_trans


class TCGA_Image_Dataset_name(data.Dataset):
    def __init__(self, data_path, imsize=(224, 224), is_sort=False):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.image_path = collect_pathology_image_paths(data_path)

        self.tr_transforms2D = get_train_transform2D(imsize)

        if is_sort:
            self.img_ids = sorted(self.image_path)
            print("sorted dataset")

        print("image sample number: ", len(self.image_path))

    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, index):
        img_path = self.image_path[index]
        image2D = load_pathology_image_pil_rgb(img_path)
        image2D = self.tr_transforms2D(image2D)
        return image2D, img_path
    
def my_collate_path(batch):
    image2D, key = zip(*batch)
    image2D = torch.stack(image2D, 0)
    
    return [image2D, key]

def get_train_transform2D(crop_size):

    tr_transforms = transforms.Compose(
        [
        # transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()
         ])

    return tr_transforms




