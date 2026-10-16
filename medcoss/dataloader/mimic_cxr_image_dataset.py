import os
import numpy as np
import torch
import torch.utils.data as data
import json
import cv2
import pydicom
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XRAY_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_xray_image_file(filename):
    return os.path.splitext(filename.lower())[1] in XRAY_IMAGE_EXTENSIONS


def load_xray_image_pil_rgb(img_path):
    """Load chest X-ray as RGB PIL Image; cv2 first, then PIL fallback."""
    gr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gr is not None and gr.size > 0:
        return Image.fromarray(gr).convert("RGB")
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(
            f"Failed to read X-ray image {img_path!r} (cv2 and PIL failed): {e}"
        ) from e


def collect_xray_image_paths(data_path, list_filename="pretrain_data_list.json"):
    """Load paths from JSON if present; else recursive scan, write JSON, return paths."""
    list_path = os.path.join(data_path, list_filename)
    if os.path.exists(list_path):
        with open(list_path, "r") as f:
            return json.load(f)["path"]
    paths = []
    for root, dirs, files in tqdm(os.walk(data_path), desc="scan xray"):
        for fname in files:
            if fname == list_filename:
                continue
            if is_xray_image_file(fname):
                paths.append(os.path.join(root, fname))
    data_list = {"path": paths}
    with open(list_path, "w") as f:
        json.dump(data_list, f, sort_keys=True, indent=4)
    return paths


# print("base_dir", BASE_DIR)
def read_from_dicom(img_path, imsize=None, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x = resize_img(x, imsize)

    img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(img)

    return img


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

class MIMIC_CXR_Image_Dataset(data.Dataset):
    def __init__(self, data_path, imsize=(224, 224), is_sort=False):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.image_path = collect_xray_image_paths(data_path)

        self.tr_transforms2D = get_train_transform2D(imsize)

        if is_sort:
            self.img_ids = sorted(self.image_path)
            print("sorted dataset")

        print("image sample number: ", len(self.image_path))

    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, index):
        img_path = self.image_path[index]
        image2D = load_xray_image_pil_rgb(img_path)
        image2D_trans = self.tr_transforms2D(image2D)
        return image2D_trans



class MIMIC_CXR_Image_Dataset_name(data.Dataset):
    def __init__(self, data_path, imsize=(224, 224), is_sort=False):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.image_path = collect_xray_image_paths(data_path)

        self.tr_transforms2D = get_train_transform2D(imsize)

        if is_sort:
            self.img_ids = sorted(self.image_path)
            print("sorted dataset")

        print("image sample number: ", len(self.image_path))

             
    def __len__(self):
        return len(self.image_path)


    def __getitem__(self, index):
        img_path = self.image_path[index]
        image2D = load_xray_image_pil_rgb(img_path)
        image2D = self.tr_transforms2D(image2D)
        return image2D, img_path
    

def my_collate_xray(batch):
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


def processing(path, out_path):
    img = Image.open(path).resize((224, 224))
    img.save(os.path.join(out_path, path.split("/")[-1]))



