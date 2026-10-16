import os
import pickle
import re
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_CAPTION_CACHE_DIR = os.path.join(BASE_DIR, "report_caption_caches")


def normalize_path_value(path_str, path_root):
    """Resolve a Path cell to an absolute path: prefer relative-to-root, then legacy MIMIC-style drop-first-segment."""
    if path_str is None or (isinstance(path_str, float) and pd.isna(path_str)):
        return ""
    p = str(path_str).strip()
    if not p:
        return p
    p_norm = os.path.normpath(p.replace("\\", os.sep).replace("/", os.sep))
    if os.path.isabs(p_norm):
        return p_norm
    root = os.path.normpath(path_root)
    cand = os.path.normpath(os.path.join(root, p_norm))
    if os.path.exists(cand):
        return cand
    parts = re.split(r"[\\/]+", p.strip())
    if len(parts) > 1:
        legacy = os.path.normpath(os.path.join(root, *parts[1:]))
        if os.path.exists(legacy):
            return legacy
    return cand


def normalize_report_paths_in_df(df, path_root, path_col="Path"):
    """In-place: normalize Path column."""
    df[path_col] = df[path_col].apply(lambda x: normalize_path_value(x, path_root))


def build_path2sent_from_df(df):
    """Build path -> sentence list from impression + findings (same token rules as original MedCoSS)."""
    sent_lens, num_sents = [], []
    path2sent = {}
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        captions = ""
        captions += str(row["impression"])
        captions += " "
        captions += str(row["findings"])
        captions = captions.replace("\n", " ")
        splitter = re.compile("[0-9]+\.")
        captions = splitter.split(captions)
        captions = [point.split(".") for point in captions]
        captions = [sent for point in captions for sent in point]

        cnt = 0
        study_sent = []
        for cap in captions:
            if len(cap) == 0:
                continue
            cap = cap.replace("\ufffd\ufffd", " ")
            tokenizer = RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(cap.lower())
            if len(tokens) <= 1:
                continue
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)
            if len(included_tokens) > 0:
                study_sent.append(" ".join(included_tokens))
            cnt += len(included_tokens)

        if cnt >= 3:
            sent_lens.append(cnt)
            num_sents.append(len(study_sent))
            path2sent[row["Path"]] = study_sent

    sent_lens = np.array(sent_lens)
    num_sents = np.array(num_sents)
    if len(sent_lens) > 0:
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )
    return path2sent


def load_or_build_path2sent_csv(captions_csv_path, df):
    """
    Cache path2sent by SHA256 of the captions CSV bytes so master vs buffer subsets never collide.
    df must use normalized Path values matching rows in captions_csv_path.
    """
    os.makedirs(REPORT_CAPTION_CACHE_DIR, exist_ok=True)
    with open(captions_csv_path, "rb") as f:
        fp = hashlib.sha256(f.read()).hexdigest()
    cache_file = os.path.join(REPORT_CAPTION_CACHE_DIR, f"{fp}.pickle")
    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    path2sent = build_path2sent_from_df(df)
    with open(cache_file, "wb") as f:
        pickle.dump(path2sent, f, protocol=2)
    print("Saved report caption cache:", cache_file)
    return path2sent


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


def get_imgs(img_path, scale, transform=None, multiscale=False):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)

    return img


class MIMIC_CXR_Report_Dataset(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.data_path = data_path
        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(os.path.join(data_path, "master.csv"))
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        normalize_report_paths_in_df(self.df, data_path)

        self.filenames, self.path2sent = self.load_text_data(split)

        self.df = self.df[self.df["split"] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

        self.mlm_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=True,
                                                         mlm_probability=0.15)

        print("report sample number:", len(self.filenames))

    def load_text_data(self, split):
        master_csv = os.path.join(self.data_path, "master.csv")
        path2sent = load_or_build_path2sent_csv(master_csv, self.df)
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, "split")
            path = getattr(row, "Path")
            if cur_split == split and path in path2sent:
                filenames.append(path)
        return filenames, path2sent

    def __len__(self):
        return len(self.filenames)

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
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        text, attention_mask = caps["input_ids"], caps["attention_mask"]
        caps_mask = self.mlm_collator(tuple(text))

        return caps_mask["input_ids"].squeeze(0), caps_mask["labels"].squeeze(0), attention_mask.squeeze(0), text.squeeze(0)


class MIMIC_CXR_Report_Dataset_name(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.data_path = data_path
        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(os.path.join(data_path, "master.csv"))
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        normalize_report_paths_in_df(self.df, data_path)

        self.filenames, self.path2sent = self.load_text_data(split)

        self.df = self.df[self.df["split"] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

        self.mlm_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=False,
                                                         mlm_probability=0.0)

        print("report sample number:", len(self.filenames))

    def load_text_data(self, split):
        master_csv = os.path.join(self.data_path, "master.csv")
        path2sent = load_or_build_path2sent_csv(master_csv, self.df)
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, "split")
            path = getattr(row, "Path")
            if cur_split == split and path in path2sent:
                filenames.append(path)
        return filenames, path2sent

    def __len__(self):
        return len(self.filenames)

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
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        text, attention_mask = caps["input_ids"], caps["attention_mask"]
        caps_mask = self.mlm_collator(tuple(text))
        return caps_mask["input_ids"].squeeze(0), caps_mask["labels"].squeeze(0), attention_mask.squeeze(0), text.squeeze(0), key


def my_collate_text(batch):
    input_ids, labels, attention_mask, text, key = zip(*batch)
    input_ids = torch.stack(input_ids, 0)
    labels = torch.stack(labels, 0)
    attention_mask = torch.stack(attention_mask, 0)
    text = torch.stack(text, 0)
    return [input_ids, labels, attention_mask, text, key]

