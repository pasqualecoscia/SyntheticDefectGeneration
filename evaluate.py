import os
import glob

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.io as io
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils_datasets import ImageDataset
from PIL.Image import Resampling
# from src.models import define_generator
from src.test_options import TestOptions
from src.utils import create_test_folders, select_input_root, normalize_images_diff
from src.model_selection import select_model

import os
import argparse
import numpy as np
import pickle
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from metrics import FID, LPIPS, LPIPS_to_train

if __name__ == '__main__':

    # Load test options
    args = TestOptions().parse()  
    # Input parser for epoch selection
    parser = argparse.ArgumentParser(description='Evaluation metrics parser')
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument('--epoch', type=int, required=True, help='epoch to evaluate')
    parser.add_argument('--dims', type=int, default=64, help='FID dims features [64, 192, 768, 2048]')
    args_input = parser.parse_args()
    
    # Set cudnn
    cudnn.benchmark = True

    # GPU check
    if torch.cuda.is_available() and args_input.cuda == True:
        device = 'cuda'
    else:
        device = 'cpu'
        print("WARNING: you should probably run with --cuda if you have a GPU support.")
        
    print(device)
    DATA_PATH = os.path.join(args.outf, args.dataset, "test", args.input_type, "A2B/", "epoch_" + str(args_input.epoch))
    DATA_PATH_REAL_B = os.path.join(args.dataroot, args.dataset, "train", "B/")
    # Read fake/real images
    list_fake_image_B, list_real_image_A, list_train_image_B = list(), list(), list()
    paths_fake_image_B, paths_real_image_A, paths_train_image_B = list(), list(), list()
    names_fake_image_B = sorted([f for f in os.listdir(DATA_PATH) if ("fake" in f.split('.')[0] and "mask" not in f.split('.')[0])])
    names_real_image_A = sorted([f for f in os.listdir(DATA_PATH) if "real" in f.split('.')[0]])
    names_train_image_B = sorted([f for f in os.listdir(DATA_PATH_REAL_B)])

    for i in range(len(names_fake_image_B)):
        ##############################  CHECK RESIZE
        p = os.path.join(DATA_PATH, names_fake_image_B[i])
        im = (Image.open(p).convert("RGB"))
        list_fake_image_B += [im.resize((args.image_size, args.image_size), Resampling.BILINEAR)]
        paths_fake_image_B.append(p)
    #im_res = (ToTensor()(list_fake_image[0]).shape[2], ToTensor()(list_fake_image[0]).shape[1])
    for i in range(len(names_real_image_A)):
        p = os.path.join(DATA_PATH, names_real_image_A[i])
        im = (Image.open(p).convert("RGB"))
        list_real_image_A += [im.resize((args.image_size, args.image_size), Resampling.BILINEAR)]
        paths_real_image_A.append(p)
    for i in range(len(names_train_image_B)):
        p = os.path.join(DATA_PATH_REAL_B, names_train_image_B[i])
        im = (Image.open(p).convert("RGB"))
        list_train_image_B += [im.resize((args.image_size, args.image_size), Resampling.BILINEAR)]
        paths_train_image_B.append(p)

    # --- Compute the metrics --- #
    with torch.no_grad():
        fid_score = FID(paths_train_image_B, paths_fake_image_B, dims=args_input.dims, device=device)
        lpips = LPIPS(list_fake_image_B)
        dist_to_tr, dist_to_tr_byimage = LPIPS_to_train(list_train_image_B, list_fake_image_B, names_fake_image_B)

    print(f"FID: {fid_score:.2f}")
    print(f"LPIPS: {lpips.item():.2f}")
    print(f"LPIPS_to_train: {dist_to_tr:.2f}")

