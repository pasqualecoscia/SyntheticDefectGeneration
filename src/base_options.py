import argparse

class BaseOptions():
    
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used for both training and test sets."""
        # saving frequencies
        parser.add_argument("--save_freq_images", default=100, type=int, help="print frequency. (default:100)")
        parser.add_argument("--save_freq", default=25, type=int, help="saving frequency. (default:25). The net will be tested on each saved parameters.")
        # Number of training epochs (used also for loading parameters during testing)
        parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
        # basic parameters
        parser.add_argument("--dataroot", type=str, default="./data", help="path to datasets. (default:./data)")
        parser.add_argument("--dataset", type=str, default="mvtec_dataset", help="dataset name. (default:`horse2zebra`) Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, selfie2anime, iphone2dslr_flower, ae_photos, ]")
        parser.add_argument("--mask", action="store_false", help="Load binary masks for defective images.")
        parser.add_argument("--outf", default="./results", help="folder for output images. (default:'./results').")
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan_mask', help='specify model type [cycle_gan|cycle_gan_mask]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        # Network layers
        parser.add_argument('--netG', type=str, default='ResNet9', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        # dataset parameters
        parser.add_argument("--image_size", type=int, default=256, help="size of the data crop (squared assumed). (default:256)")
        # additional parameters
        parser.add_argument("--cuda", action="store_true", help="Enables cuda")
        # Mask threshold
        parser.add_argument("--thrs_mask", type=float, default=0.2, help="Threshold for obtaining defect segmentation mask from raw logits")
        self.initialized = True

        return parser

    def gather_options(self):
        """Initialize parser with basic options (only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(description="INFECT: Defects Blending for Industrial Products")
            parser = self.initialize(parser)        
        
        # save and return the parser
        self.parser = parser
        return parser.parse_args()


    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>20}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # utils.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        """ Parse and print options """
        opt = self.gather_options()
        
        self.print_options(opt)

        self.opt = opt
        return self.opt
