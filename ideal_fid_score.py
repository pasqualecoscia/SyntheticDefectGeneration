from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
import os
import torch
import torchvision.io as io
import argparse

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

parser = argparse.ArgumentParser(
    description="Adversarial Defect Synthesis - PyTorch")

parser.add_argument("--dataroot", type=str, default="./data", help="path to datasets. (default:./data)")
parser.add_argument("--dataset", type=str, default="mvtec_dataset", help="dataset name. (default:`horse2zebra`) Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, selfie2anime, iphone2dslr_flower, ae_photos, ]")

args = parser.parse_args()

# Compute FID score
fid = FrechetInceptionDistance(feature=2048)

PATH_1= os.path.join(args.dataroot, args.dataset, "train/B")
PATH_2 = os.path.join(args.dataroot, args.dataset, "test/B")

# Resize the image with given size
transform = transforms.Resize(256)

images_1 = read_images(PATH_1, transform)
images_2 = read_images(PATH_2, transform)

fid.update(images_1, real=True)
fid.update(images_2, real=False)
print(f"FID score: {fid.compute():.2f}.")