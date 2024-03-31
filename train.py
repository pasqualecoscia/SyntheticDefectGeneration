import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from src.utils_datasets import ImageDataset
from src.utils import create_train_folders
from src.model_selection import select_model
from src.train_options import TrainOptions

if __name__ == '__main__':

    # Load train options
    args = TrainOptions().parse()   # get training options

    # Create training folders
    create_train_folders(args)

    # Set cudnn
    cudnn.benchmark = True

    # GPU check
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Dataset
    # Normalize between [-1, 1]
    mean = (.5, .5, .5)
    std = (.5, .5, .5)

    DATASET_ROOT = os.path.join(args.dataroot, args.dataset)
    
    # Apply same transformation to image and mask (if provided)
    if args.mask:
        additional_targets = {
        'imageB': 'image',
        'maskB': 'mask',            
        }
    else:
        additional_targets = {
        'imageB': 'image',
        }

    transform = A.Compose([
        A.Resize(int(args.image_size * 1.12), int(args.image_size * 1.12), interpolation=cv2.INTER_CUBIC),
        A.RandomCrop(args.image_size, args.image_size, p=1), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.ShiftScaleRotate (shift_limit=0.05, scale_limit=0.2, rotate_limit=10, interpolation=1, \
        #     border_mode=1, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, rotate_method='largest_box', always_apply=False, p=0.5),
        A.ElasticTransform (alpha=1, sigma=2, alpha_affine=0.5, interpolation=cv2.INTER_CUBIC, \
           border_mode=cv2.BORDER_REFLECT, value=None, mask_value=0, always_apply=False, approximate=False, \
               same_dxdy=False, p=0.5),
        #A.RandomBrightnessContrast(p=0.2),
        #A.RandomContrast(p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ],
    additional_targets=additional_targets
    )


    dataset = ImageDataset(root=DATASET_ROOT,
                        # transform_mask=transforms.Compose([
                        #     transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                        #     transforms.RandomCrop(args.image_size),
                        #     transforms.RandomHorizontalFlip(),
                        #     transforms.ToTensor(),
                        #     ]),
                        # transform_image = transforms.Normalize(mean=mean, std=std),
                        transform = transform,
                        unaligned=True,
                        mask=args.mask)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Create model
    model = select_model(args)

    model.create_network()

    # Resume training if paths are provided
    model.resume_training()

    # Define losses
    model.define_losses()

    # Set optimizer
    model.set_optimizer()

    # Set decay LR
    model.set_decayLR()

    labels = {'real_label': 1, 'fake_label': 0}

    for epoch in range(1, args.epochs+1):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress_bar:

            # Prepare input data
            inputs = (data, labels)
            inputs = model.prepare_inputs(inputs)

            (output, losses) = model.compute_loss_and_update(inputs)

            if i == 0:
                model.save_training_progress(inputs, mean, std, epoch)

            # Print progress bar info
            progress_bar.set_description(model.set_description(losses, epoch, args.epochs, len(dataloader), i))

            # Save images at specific frequencies
            if i % args.save_freq_images == 0:
                model.save_training_images(inputs, output, epoch, i)

        if epoch % (args.save_freq) == 0:
            # Save checkpoints
            model.save_parameters(epoch)

        # Update learning rates
        model.update_learning_rates()
