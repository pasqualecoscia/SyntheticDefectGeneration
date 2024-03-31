from src.cycle_gan import CycleGAN
from src.cycle_gan_mask import CycleGAN_Mask
from src.gan_mask import GAN_Mask

def select_model(args):
    """ Select model """
    if args.model == 'cycle_gan':
        model = CycleGAN(args)
    elif args.model == 'cycle_gan_mask':
        model = CycleGAN_Mask(args)
    #TODO:else
    return model