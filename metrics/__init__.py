import torch
from torchvision.transforms import ToTensor
from .FID.fid_score import calculate_fid_given_paths
from .mIoU.main import compute_miou
from .LPIPS.models import PerceptualLoss

p_model = PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

def FID(list_real_image, list_fake_image, dims=2048, device='cpu'):
    """ Compute FID score between two sets of images"""
    
    device = torch.device(device)

    fid_value = calculate_fid_given_paths(paths=[list_real_image, list_fake_image],
        batch_size=1,
        device=device,
        dims=dims,
        num_workers=1)    
    
    return fid_value


def LPIPS(list_fake_image):
    """
    Compute average LPIPS between pairs of fake images
    """
    dist_diversity = 0
    count = 0
    lst_im = list()
    # --- unpack images --- #
    for i in range(len(list_fake_image)):
        lst_im.append(ToTensor()(list_fake_image[i]).unsqueeze(0))
    # --- compute LPIPS between pairs of images --- #
    for i in range(len(lst_im))[:100]:
        for j in range(i + 1, len(lst_im))[:100]:
            dist_diversity += p_model.forward(lst_im[i], lst_im[j])
            count += 1
    return dist_diversity/count


def LPIPS_to_train(list_real_image, list_fake_image, names_fake_image):
    """
    For each fake image find the LPIPS to the closest training image
    """
    dist_to_real_dict = dict()
    ans1 = 0
    count = 0
    lst_real, list_fake = list(), list()
    # --- unpack images --- #
    for i in range(len(list_fake_image)):
        list_fake.append(ToTensor()(list_fake_image[i]).unsqueeze(0))
    for i in range(len(list_real_image)):
        lst_real.append(ToTensor()(list_real_image[i]).unsqueeze(0))
    # --- compute average minimum LPIPS from a fake image to real images --- #
    for i in range(len(list_fake)):
        tens_im1 = list_fake[i]
        cur_ans = list()
        for j in range(len(lst_real)):
            tens_im2 = lst_real[j]
            dist_to_real = p_model.forward(tens_im1, tens_im2)
            cur_ans.append(dist_to_real)
        cur_min = torch.min(torch.Tensor(cur_ans))
        dist_to_real_dict[names_fake_image[i]] = float(cur_min.detach().cpu().item())
        ans1 += cur_min
        count += 1
    ans = ans1 / count
    return ans, dist_to_real_dict

def mIoU(path_real_images, names_real_image, path_real_masks, names_real_masks,
             exp_folder, names_fake_image, names_fake_masks, im_res):
    """
    Train a simple UNet on fake (real) images&masks, test on real (fake) images&masks.
    Report mIoU and segmentation accuracy for the whole sets (fake->real and  real->fake) as well as
    individual scores for each fake image
    """
    metrics_tensor, results, results_acc = compute_miou(path_real_images, names_real_image, path_real_masks, names_real_masks,
                                                        exp_folder, names_fake_image, names_fake_masks, im_res)
    return metrics_tensor, results, results_acc
