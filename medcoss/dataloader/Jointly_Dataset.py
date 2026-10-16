import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm
from transformers import BertTokenizer
import cv2
import pydicom
from PIL import Image
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizerFast,
    RobertaTokenizerFast
)
import json
from torchvision import transforms
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("base_dir", BASE_DIR)
import nibabel as nib
import random
import shutil

from dataloader.mimic_cxr_report_dataset import (
    load_or_build_path2sent_csv,
    normalize_report_paths_in_df,
)
from dataloader.mimic_cxr_image_dataset import collect_xray_image_paths, load_xray_image_pil_rgb
from dataloader.TCGA_dataset import collect_pathology_image_paths, load_pathology_image_pil_rgb


class DataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5))
            ]
        else:
            data_transforms = [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)

def get_imgs(img_path, scale, transform=None, multiscale=False):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
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


def get_imgs(img_path, scale, transform=None, multiscale=False):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)

    return img

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


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def get_train_transform2D(crop_size):

    tr_transforms = transforms.Compose(
        [
        # transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()
         ])

    return tr_transforms


class Buffer_Dataset(data.Dataset):
    def __init__(self, data_path_text, data_path_xray, data_path_ct, data_path_mr, data_path_path, split="train", transform_text=None, data_pct=1.0,
                 imsize=224, max_words=112, crop_size=(16, 192, 192), batch_size=128, buffer_data=["1D_text"], task_data=[""],num_center=None,buffer_ratio=None,exp_name=None,
                 buffer_file_path=None, file_copy_path=None):
        super().__init__()
        self.text_filenames, self.xray_image_path, self.ct_files, self.mr_files, self.path_image_path = [], [], [], [], []
        self.batch_size = batch_size
        self.imsize = imsize

        # MIMIC_CXR_Report
        if "1D_text" in buffer_data:
            if not os.path.exists(data_path_text):
                raise RuntimeError(f"{data_path_text} does not exist!")
            file_name = f'1D_text_{num_center}_{buffer_ratio}_{exp_name}.csv'
            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))
            print("load data from ", os.path.join(buffer_file_path, file_name))

            self.text_transform = transform_text

            self.df = pd.read_csv(os.path.join(buffer_file_path, file_name))
            self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
            normalize_report_paths_in_df(self.df, data_path_text)

            captions_csv_path = os.path.join(buffer_file_path, file_name)
            self.text_filenames, self.path2sent = self.load_text_data(split, captions_csv_path)

            self.df = self.df[self.df["split"] == split]
            if data_pct != 1.0 and split == "train":
                self.df = self.df.sample(frac=data_pct, random_state=42)
            self.df.reset_index(drop=True, inplace=True)

            self.tokenizer = BertTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT") #path of Bio_ClinicalBERT
            self.max_words = max_words

            self.mlm_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
            print("report sample number:", len(self.text_filenames))

        elif "1D_text" in task_data:
            file_name = "master.csv"

            self.text_transform = transform_text
            self.imsize = imsize
            self.df = pd.read_csv(os.path.join(data_path_text, file_name))
            self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
            normalize_report_paths_in_df(self.df, data_path_text)

            captions_csv_path = os.path.join(data_path_text, file_name)
            self.text_filenames, self.path2sent = self.load_text_data(split, captions_csv_path)

            self.df = self.df[self.df["split"] == split]
            if data_pct != 1.0 and split == "train":
                self.df = self.df.sample(frac=data_pct, random_state=42)
            self.df.reset_index(drop=True, inplace=True)

            self.tokenizer = BertTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT")
            self.max_words = max_words

            self.mlm_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=True,
                                                             mlm_probability=0.15)

            print("report sample number:", len(self.text_filenames))

        elif task_data == "":
            file_name = f'1D_text_{num_center}_{buffer_ratio}_{exp_name}.csv'
            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))

        # MIMIC_CXR_Image
        if "2D_xray" in buffer_data:

            file_name = f'2D_xray_{num_center}_{buffer_ratio}_{exp_name}.json'

            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))
            print("load data from ", os.path.join(buffer_file_path, file_name))
            if not os.path.exists(data_path_xray):
                raise RuntimeError(f"{data_path_xray} does not exist!")

            # find all images
            self.xray_image_path = []
            self.xray_image_path = load_json(os.path.join(buffer_file_path, file_name))["path"]
            self.xray_tr_transforms2D = get_train_transform2D(imsize)

            print("xray sample number: ", len(self.xray_image_path))

        elif "2D_xray" in task_data:
            if not os.path.exists(data_path_xray):
                raise RuntimeError(f"{data_path_xray} does not exist!")

            self.xray_image_path = collect_xray_image_paths(data_path_xray)
            self.xray_tr_transforms2D = get_train_transform2D(imsize)

            print("xray sample number: ", len(self.xray_image_path))

        elif task_data == "":
            file_name = f'2D_xray_{num_center}_{buffer_ratio}_{exp_name}.json'

            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))

        # 3D CT
        if "3D_CT" in buffer_data:
            file_name = f'3D_CT_{num_center}_{buffer_ratio}_{exp_name}.txt'
            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))
            print("load data from ", os.path.join(buffer_file_path, file_name))
            self.ct_data_path = data_path_ct
            self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = crop_size
            self.ct_img_ids = [i_id.strip().split() for i_id in
                               open(os.path.join(buffer_file_path, file_name))]
            self.ct_files = []
            for nii_name in self.ct_img_ids:
                ct_img_file = os.path.join(self.ct_data_path, nii_name[0])
                self.ct_files.append({
                    "img": ct_img_file.replace("DL_patches_v2", "DL_patches_v2_resize"),
                    "name": nii_name[0]
                })
            print('SSL CT: {} images are loaded!'.format(len(self.ct_files)))
            self.ct_transformer = MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label")

        elif "3D_CT" in  task_data:
            self.ct_data_path = data_path_ct
            self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = crop_size
            self.ct_img_ids = [i_id.strip().split() for i_id in open(os.path.join(data_path_ct, "SSL_data_deeplesion.txt"))]

            self.ct_files = []
            for nii_name in self.ct_img_ids:
                ct_img_file = os.path.join(self.ct_data_path, nii_name[0])
                self.ct_files.append({
                    "img": ct_img_file.replace("DL_patches_v2", "DL_patches_v2_resize"),
                    "name": nii_name[0]
                })
            print('SSL CT: {} images are loaded!'.format(len(self.ct_files)))
            self.ct_transformer = MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label")

        elif task_data == "":
            file_name = f'3D_CT_{num_center}_{buffer_ratio}_{exp_name}.txt'
            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))

        # 3D MR
        if "3D_MR" in buffer_data:
            file_name = f'3D_MR_{num_center}_{buffer_ratio}_{exp_name}.txt'
            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))
            print("load from", os.path.join(buffer_file_path, file_name))
            self.mr_data_path = data_path_mr
            self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = crop_size
            self.mr_img_ids = [i_id.strip().split() for i_id in open(os.path.join(buffer_file_path, file_name ))]

            self.mr_files = []
            for nii_name in self.mr_img_ids:
                mr_img_file = os.path.join(self.mr_data_path, nii_name[0])
                self.mr_files.append({
                    "img": mr_img_file,
                    "name": nii_name[0]
                })
            print('SSL MR: {} images are loaded!'.format(len(self.mr_files)))
            self.mr_transformer = MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label")

        elif "3D_MR" in task_data:
            self.mr_data_path = data_path_mr
            self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = crop_size
            self.mr_img_ids = [i_id.strip().split() for i_id in open(os.path.join(data_path_mr, "SSL_data_ADNI.txt"))]

            self.mr_files = []
            for nii_name in self.mr_img_ids:
                mr_img_file = os.path.join(self.mr_data_path, nii_name[0])
                self.mr_files.append({
                    "img": mr_img_file,
                    "name": nii_name[0]
                })
            print('SSL MR: {} images are loaded!'.format(len(self.mr_files)))
            self.mr_transformer = MirrorTransform(axes=(0, 1, 2), data_key="image", label_key="label")

        elif task_data == "":
            file_name = f'3D_MR_{num_center}_{buffer_ratio}_{exp_name}.txt'
            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))

        # 2D pathology
        if "2D_path" in buffer_data:
            file_name = f'2D_path_{num_center}_{buffer_ratio}_{exp_name}.json'

            shutil.copyfile(os.path.join(buffer_file_path, file_name), os.path.join(file_copy_path, file_name))
            print("load data from ", os.path.join(buffer_file_path, file_name))
            if not os.path.exists(data_path_path):
                raise RuntimeError(f"{data_path_path} does not exist!")

            # find all images
            self.path_image_path = []
            self.path_image_path = load_json(os.path.join(buffer_file_path, file_name))["path"]
            self.path_tr_transforms2D = get_train_transform2D(imsize)

            print("path sample number: ", len(self.path_image_path))

        elif "2D_path" in task_data:
            if not os.path.exists(data_path_path):
                raise RuntimeError(f"{data_path_path} does not exist!")

            self.path_image_path = collect_pathology_image_paths(data_path_path)

            self.path_tr_transforms2D = get_train_transform2D(imsize)
            print("pathology sample number: ", len(self.path_image_path))

        elif task_data == "":
            pass


        self.files = []
        # for _ in range(len(self.text_filenames)//(batch_size*num_task)):
        self.files.extend([0] * (len(self.text_filenames) // (batch_size)))
        self.files.extend([1] * (len(self.xray_image_path) // (batch_size)))
        self.files.extend([2] * (len(self.ct_files) // (batch_size)))
        self.files.extend([3] * (len(self.mr_files) // (batch_size)))
        self.files.extend([4] * (len(self.path_image_path) // (batch_size)))


        print("iteration each dataset: total: {}, Text: {}, Xray: {}, CT: {}, MR: {}, Path: {}".format(len(self.files),
              len(self.text_filenames) // (batch_size), len(self.xray_image_path) // (batch_size), len(self.ct_files) // (
               batch_size), len(self.mr_files) // (batch_size), len(self.path_image_path) // (batch_size)))


    def load_text_data(self, split, captions_csv_path):
        path2sent = load_or_build_path2sent_csv(captions_csv_path, self.df)
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, "split")
            path = getattr(row, "Path")
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def __len__(self):
        return len(self.files)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def __getitem__(self, index):
        data_loader_index = self.files[index]
        task_id = torch.Tensor([data_loader_index])

        if data_loader_index == 0:
            selected_keys = np.random.choice(self.text_filenames, self.batch_size, True, None)
            # print("dataset", selected_keys)
            input_ids = torch.zeros(size=(self.batch_size, self.max_words))
            labels = torch.zeros(size=(self.batch_size, self.max_words))
            attention_mask = torch.zeros(size=(self.batch_size, self.max_words))
            for j, key in enumerate(selected_keys):

                caps, cap_len = self.get_caption(key)
                text, atte_mask = caps["input_ids"], caps["attention_mask"]
                caps = self.mlm_collator(tuple(text))
                input_ids[j] = caps["input_ids"].squeeze(0)
                labels[j] = caps["labels"].squeeze(0)
                attention_mask[j] = atte_mask.squeeze(0)

            return input_ids, labels, attention_mask, task_id
        
        elif data_loader_index == 1:
            selected_keys = np.random.choice(self.xray_image_path, self.batch_size, True, None)
            # print("dataset", selected_keys)
            image2Ds = torch.zeros(size=(self.batch_size, 3, self.imsize, self.imsize))
            for j, img_path in enumerate(selected_keys):

                image2D = load_xray_image_pil_rgb(img_path)
                image2D_trans = self.xray_tr_transforms2D(image2D)
                image2Ds[j] = image2D_trans
            return image2Ds, task_id

        elif data_loader_index == 2:
            selected_keys = np.random.choice(self.ct_files, self.batch_size, True, None)
            # print("dataset", selected_keys)
            image3Ds = np.zeros(shape=(self.batch_size, 1, self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w))
            for j, datafiles in enumerate(selected_keys):

                imageNII = nib.load(datafiles["img"])
                image = imageNII.get_fdata()
                image = image / 1024.

                image = image[np.newaxis, :][np.newaxis, :]
                image = image.transpose((0, 1, 4, 2, 3))

                data_dict = {'image': image, 'label': None}

                if random.randint(0, 1) == 0:
                    data_dict = self.ct_transformer(**data_dict)
                image3Ds[j] = data_dict["image"][0]

            return image3Ds, task_id
        
        elif data_loader_index == 3:
            selected_keys = np.random.choice(self.mr_files, self.batch_size, True, None)
            # print("dataset", selected_keys)
            image3Ds = np.zeros(shape=(self.batch_size, 1, self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w))
            for j, datafiles in enumerate(selected_keys):
                # read nii file
                # image = np.load(datafiles["img"])
                imageNII = nib.load(datafiles["img"])
                image = imageNII.get_fdata()

                image = image[np.newaxis, :][np.newaxis, :]
                image = image.transpose((0, 1, 4, 2, 3))

                data_dict = {'image': image, 'label': None}

                if random.randint(0, 1) == 0:
                    data_dict = self.mr_transformer(**data_dict)
                image3Ds[j] = data_dict["image"][0]

            return image3Ds, task_id

        elif data_loader_index == 4:
            selected_keys = np.random.choice(self.path_image_path, self.batch_size, True, None)
            # print("dataset", selected_keys)
            image2Ds = torch.zeros(size=(self.batch_size, 3, self.imsize, self.imsize))
            for j, img_path in enumerate(selected_keys):

                image2D = load_pathology_image_pil_rgb(img_path)
                image2D_trans = self.path_tr_transforms2D(image2D)
                image2Ds[j] = image2D_trans
            return image2Ds, task_id

