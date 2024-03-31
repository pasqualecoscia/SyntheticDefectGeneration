import os
import glob

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.io as io
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils_datasets import ImageDataset

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
# from src.models import define_generator
from src.test_options import TestOptions
from src.utils import create_test_folders, select_input_root, normalize_images_diff
from src.model_selection import select_model

def remove_duplicates(path, num_files):
    '''
    Remove duplicates if images are not aligned and each domain has a different number of files
    '''
    files_list = sorted(glob.glob(path + "*.*"))
    files_list = files_list[num_files:]
    for p in files_list:
        os.remove(p)

def read_images(path, transform):
    '''
    Read images in path and applies transformation 'transform'.

    Input
    ------
        path
        transform: torchvision.transforms.Resize
    
    Output
    ------
        batch: torch.Tensor (N x 3 x H x W)
    '''
    
    batch_size = len(os.listdir(path))
    batch = torch.zeros(batch_size, 3, transform.size, transform.size, dtype=torch.uint8)
    for i, filename in enumerate(os.listdir(path)):
        batch[i] = transform(io.read_image(os.path.join(path, filename)))

    return batch

if __name__ == '__main__':

    # Load test options
    args = TestOptions().parse()  
    
    # Set cudnn
    cudnn.benchmark = True

    # GPU check
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Dataset
    mean_ = (.5, .5, .5)
    std_ = (.5, .5, .5)
    # Select input type
    ROOT_PATH = select_input_root(args)

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
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=mean_, std=std_),
        ToTensorV2()
        ],
    additional_targets=additional_targets
    )
    
    dataset = ImageDataset(root=ROOT_PATH,
                    # transform=transforms.Compose([
                    #     transforms.Resize((args.image_size, args.image_size)),
                    #     transforms.ToTensor(),
                    #     transforms.Normalize(mean=mean, std=std)
                    # ]),
                    transform=transform,
                    mode="test",
                    mask=args.mask)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Test every save_freq iteration
    epochs = list(range(args.save_freq, args.epochs + 1, args.save_freq))
    for epoch in epochs:
        # Create test folders
        create_test_folders(args, epoch)

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        # Create models
        model = select_model(args)
        model.create_network(mode='test')

        # Load weights
        model.load_weights(epoch)
        
        # Set evaluation mode
        model.set_eval_mode()

        # Initialize metrics
        model.metrics_initialization()

        # Normalization values
        std = torch.tensor(std_, device=model.device).view(3, 1, 1)
        mean = torch.tensor(mean_, device=model.device).view(3, 1, 1)

        for i, data in progress_bar:
            labels = {'real_label': 1, 'fake_label': 0}
            # Prepare input data
            inputs = (data, labels)
            inputs = model.prepare_inputs(inputs)

            # Model testing
            output = model.test(inputs)

            # Normalize images and computer differences for saving images
            norm_output = normalize_images_diff(output, mean, std)       
            model.save_test_images(output, norm_output, i, epoch)

            # Evaluate metrics
            model.metrics_evaluation(norm_output, output["fake_mask_B"])

            progress_bar.set_description(f"Processing images {i + 1} of {len(dataloader)}")


        # Print metrics
        model.print_evaluation_metrics()
