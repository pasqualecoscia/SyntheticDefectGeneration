import os
import numpy as np
import cv2 

def create_train_folders(args):
    """ Create train folders """

    # Create folder for output images
    try:
        os.makedirs(args.outf)
        print('Output folder created!')
    except OSError:
        print('WARNING: Output folder already created or problem encountered!')
    
    # Create folder for saving model's weights
    try:
        os.makedirs(args.weightsf)
        print('Weights folder created!')
    except OSError:
        print('WARNING: Weights folder already created or problem encountered!')

    try:
        os.makedirs(os.path.join(args.weightsf, args.dataset))
        print(f'Weights folder for dataset {args.dataset} created!')
    except OSError:
        print(f'WARNING: Weights folder for dataset {args.dataset} already created or problem encountered!')

    # Images folders    
    try:
        os.makedirs(os.path.join(args.outf, args.dataset, "train", "A2B"))
        print('A2B folder created!')
        os.makedirs(os.path.join(args.outf, args.dataset, "train", "B2A"))
        print('B2A folder created!')
    except OSError:
        print('WARNING: Train A2B or Train B2A already created or problem encountered!')
    
    # Training progress folders
    try:
        os.makedirs(os.path.join(args.outf, args.dataset, "train_progress/A2B"))
        print('Training progress A2B folder created!')
        os.makedirs(os.path.join(args.outf, args.dataset, "train_progress/B2A"))
        print('Training progress B2A folder created!')
    except OSError:
        print('WARNING: Training progress folders already created or problem encountered!')

def create_test_folders(args, epoch):
    """ Create test folders """
    # Images folders    
    try:
        os.makedirs(os.path.join(args.outf, args.dataset, "test", args.input_type, "A2B", "epoch_" + str(epoch)))
        print('A2B folder created!')
        os.makedirs(os.path.join(args.outf, args.dataset, "test", args.input_type, "B2A", "epoch_" + str(epoch)))
        print('B2A folder created!')
    except OSError:
        print('WARNING: Test A2B or Test B2A already created or problem encountered!')

def select_input_root(args):
    """ Select input root """
    if args.input_type == 'standard':  # no-modified test set
        ROOT_PATH = os.path.join(args.dataroot, args.dataset)
    elif args.input_type == 'random': # random noise input
        ROOT_PATH = os.path.join(args.dataroot, 'random')
    elif args.input_type == 'other_products': # images of other products
        ROOT_PATH = os.path.join(args.dataroot, 'other_products')
    elif args.input_type == 'checkboard': # images of checkboard patterns
        ROOT_PATH = os.path.join(args.dataroot, 'checkboard')
    elif args.input_type == 'gradient': # images of checkboard patterns
        ROOT_PATH = os.path.join(args.dataroot, 'gradient')        
    return ROOT_PATH

def normalize_images_diff(output, mean, std):
    """ Normalize input images and computes also differences between real and fake images"""
        
    real_image_A = output['real_image_A']
    real_image_B = output['real_image_B']
    fake_image_A = output['fake_image_A']
    fake_image_B = output['fake_image_B']

    # Transform [-1, 1] -> [0, 1]
    real_image_A = real_image_A * std + mean
    real_image_B = real_image_B * std + mean
    fake_image_A = fake_image_A * std + mean
    fake_image_B = fake_image_B * std + mean

    # Transform [0, 1] -> [0, 255]
    real_image_A = np.moveaxis(real_image_A.squeeze(0).cpu().numpy()*255, 0, -1).astype(np.uint8)
    real_image_B = np.moveaxis(real_image_B.squeeze(0).cpu().numpy()*255, 0, -1).astype(np.uint8)
    fake_image_A = np.moveaxis(fake_image_A.squeeze(0).cpu().numpy()*255, 0, -1).astype(np.uint8)
    fake_image_B = np.moveaxis(fake_image_B.squeeze(0).cpu().numpy()*255, 0, -1).astype(np.uint8)

    # Absolute differences
    diff_A2B = cv2.absdiff(real_image_A, fake_image_B) # -> np.abs(img1 - img2), same as PIL.ImageChops.difference(im1, im2)
    diff_B2A = cv2.absdiff(real_image_B, fake_image_A) # -> np.abs(img1 - img2)

    # Convert to grayscale image
    diff_A2B = cv2.cvtColor(diff_A2B, cv2.COLOR_RGB2GRAY)
    diff_B2A = cv2.cvtColor(diff_B2A, cv2.COLOR_RGB2GRAY)

    out = {
        'real_image_A': real_image_A,
        'real_image_B': real_image_B,
        'fake_image_A': fake_image_A,
        'fake_image_B': fake_image_B,
        'diff_A2B': diff_A2B,
        'diff_B2A': diff_B2A
    }
    return out