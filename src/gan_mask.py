import functools
import torch
from src.base_model import BaseModel
from src.models_mask import *
from torch.nn import init
import itertools
from src.utils_model import DecayLR, AdversarialLoss
from src.utils_datasets import ReplayBuffer
import os 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from src.utils import normalize_images_diff
from torchsummary import summary

class GAN_Mask(BaseModel):
    """Create a model given the option."""

    def __init__(self, args):
        super(GAN_Mask, self).__init__(args)


        # Set replay buffer
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def create_network(self, mode='train'):
        """ Create network """

        self.netG_A2B = self.create_generator(self.args.input_nc, self.args.output_nc, self.args.ngf, self.args.netG, 
                    self.device, use_dropout=False, init_type=self.args.init_type, init_gain=self.args.init_gain)
        if mode == 'train':

            self.netD_B = self.create_discriminator(self.args.input_nc, self.args.ndf, self.args.netD, self.device,
                    n_layers_D=self.args.n_layers_D, init_type=self.args.init_type, init_gain=self.args.init_gain)
            
            self.netD_fit = self.create_discriminator(self.args.input_nc + 1, self.args.ndf, self.args.netD, self.device,
                    n_layers_D=self.args.n_layers_D + 3, init_type=self.args.init_type, init_gain=self.args.init_gain)                    
      

    def resume_training(self):
        """ Resume training for each network component if path is provided """     
        if self.args.netG_A2B != "":
            self.netG_A2B.load_state_dict(torch.load(self.args.netG_A2B))
            print('netG_A2B weights loaded!')
        if self.args.netD_B != "":
            self.netD_B.load_state_dict(torch.load(self.args.netD_B))
            print('netD_B weights loaded!')
        if self.args.netD_fit != "":
            self.netD_fit.load_state_dict(torch.load(self.args.netD_fit))
            print('netD_fit weights loaded!')

    def load_weights(self):
        """ Load weights for testing """
        self.netG_A2B.load_state_dict(torch.load(os.path.join("weights", str(self.args.dataset), "netG_A2B.pth")))

    def set_eval_mode(self):
        """ Set network in evaluation mode for testing """
        self.netG_A2B.eval()


    def define_losses(self):
        """ Define loss functions """
        # define loss function (adversarial_loss) and optimizer
        self.identity_loss = torch.nn.L1Loss().to(self.device)
        self.adversarial_loss = AdversarialLoss(self.device) #torch.nn.MSELoss().to(device) #LSGAN loss
    
    def set_optimizer(self):
        """ Set optimizer """
    
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(),
                                    lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        self.optimizer_D_fit = torch.optim.Adam(self.netD_fit.parameters(), lr=self.args.lr, betas=(self.args.beta_1, self.args.beta_2))
        
    def set_decayLR(self):
        self.lr_lambda = DecayLR(self.args.epochs, 0, self.args.decay_epochs).step
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_fit = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_fit, lr_lambda=self.lr_lambda)

    def prepare_inputs(self, inputs):
        """ Prepare inputs for the model"""
        # get batch size data
        (data, labels) = inputs
        real_image_A = data["A"].to(self.device)
        real_image_B = data["B"].to(self.device)
        real_image_B_mask = data["B_mask"].to(self.device).float()
        
        # Mask normalization (only if segmentation is present -> augmentation can cut out segmentation)
        if real_image_B_mask.max():    
            real_image_B_mask /= real_image_B_mask.max()
        
        batch_size = real_image_A.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size, 1), labels['real_label'], device=self.device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), labels['fake_label'], device=self.device, dtype=torch.float32)
        
        inputs = {
            'real_image_A' : real_image_A, 
            'real_image_B' : real_image_B, 
            'real_image_B_mask' : real_image_B_mask, 
            'real_label' : real_label, 
            'fake_label' : fake_label
        }

        return inputs

    def compute_loss_and_update(self, inputs):
        """ Compute model losses and update it """

        # Prepare inputs
        real_image_A= inputs['real_image_A']
        real_image_B = inputs['real_image_B']
        real_label = inputs['real_label']
        fake_label = inputs['fake_label']
        real_image_B_mask = inputs['real_image_B_mask']

        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Set G_A and G_B's gradients to zero
        self.optimizer_G.zero_grad()

        # Identity loss
        (identity_image_B, _, _) = self.netG_A2B(real_image_B)
        loss_identity_B = self.identity_loss(identity_image_B, real_image_B)

        # GAN loss
        (fake_image_B, fake_mask_B, out_concat_B) = self.netG_A2B(real_image_A)
        fake_output_B = self.netD_B(fake_image_B)
        loss_GAN_A2B = self.adversarial_loss(fake_output_B, target_is_real=True) 

        y_hat = torch.cat((fake_image_B, fake_mask_B.unsqueeze(1)), dim = 1)
        y_hat_output = self.netD_fit(y_hat)
        y = torch.cat((real_image_B, real_image_B_mask.unsqueeze(1)), dim = 1)

        loss_GAN_fit = self.adversarial_loss(y_hat_output, target_is_real=True)
        
        # Combined loss and calculate gradients
        errG = loss_identity_B * self.args.lambda_identity_B + loss_GAN_A2B * self.args.lambda_GAN_A2B + \
                        loss_GAN_fit * self.args.lambda_GAN_fit
        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        self.optimizer_G.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        self.optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = self.netD_B(real_image_B)
        errD_real_B = self.adversarial_loss(real_output_B, target_is_real=True) #adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_image_B_buffer = self.fake_B_buffer.push_and_pop(fake_image_B)
        fake_output_B = self.netD_B(fake_image_B_buffer.detach())
        errD_fake_B = self.adversarial_loss(fake_output_B, target_is_real=False) #adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_B = (errD_real_B + errD_fake_B) / 2

        # Calculate gradients for D_B
        errD_B.backward()
        # Update D_B weights
        self.optimizer_D_B.step()     

        ##############################################
        # (4) Update D mask network: Discriminator D_mask
        ##############################################
        
        # Set D_mask gradients to zero
        self.optimizer_D_fit.zero_grad()

        # Real fit 
        real_output_y = self.netD_fit(y.detach())
        errD_real_fit = self.adversarial_loss(real_output_y, target_is_real=True) #adversarial_loss(real_output_B, real_label)

        # Fake fit 
        fake_output_y = self.netD_fit(y_hat.detach())
        errD_fake_fit = self.adversarial_loss(fake_output_y, target_is_real=False) #adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_fit = (errD_real_fit + errD_fake_fit) / 2

        # Calculate gradients for D_B
        errD_fit.backward()
        # Update D_B weights
        self.optimizer_D_fit.step()     

        losses = {
            'errD_B': errD_B, 
            'errD_fit': errD_fit,
            'errG': errG, 
            'loss_identity_B': loss_identity_B, 
            'loss_GAN_A2B': loss_GAN_A2B, 
            'loss_GAN_fit': loss_GAN_fit
            }
        
        output = {
            'fake_image_B': fake_image_B,
            'fake_mask_B': fake_mask_B,
            }

        return (output, losses)

    def set_description(self, losses, epoch, num_epochs, len_dataloader, n_iter):
        """ Define progress bar string output """
        s = f"[{epoch}/{num_epochs - 1}][{n_iter}/{len_dataloader - 1}] " + \
        f"Loss_D: {(losses['errD_B']).item():.4f} " + \
        f"Loss_D_fit: {(losses['errD_fit']).item():.4f} " + \
        f"Loss_G: {losses['errG'].item():.4f} " + \
        f"loss_G_GAN: {(losses['loss_GAN_A2B']).item():.4f} " + \
        f"Loss_G_identity: {(losses['loss_identity_B']).item():.4f} " + \
        f"loss_GAN_fit: {(losses['loss_GAN_fit']).item():.4f}"
            
        return s

    def update_learning_rates(self):
        """ Update learning rates """
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_B.step()
        self.lr_scheduler_D_fit.step()

    def save_parameters(self, epoch=None):
        """ Save network parameters """
        # do check pointing
        if epoch is not None:
            s = f"_epoch_{epoch}" # Intermediate checkpoint
        else:
            s = "" # Last checkpont
        torch.save(self.netG_A2B.state_dict(), f"{self.args.weightsf}/{self.args.dataset}/netG_A2B" + s + ".pth")
        torch.save(self.netD_B.state_dict(), f"{self.args.weightsf}/{self.args.dataset}/netD_B" + s + ".pth")        
        torch.save(self.netD_fit.state_dict(), f"{self.args.weightsf}/{self.args.dataset}/netD_fit" + s + ".pth")        

    def test(self, inputs):
        """ Test model """
        # Apply generators
        real_image_A= inputs['real_image_A']
        real_image_B = inputs['real_image_B']
        real_image_B_mask = inputs['real_image_B_mask']
        real_label = inputs['real_label']
        fake_label = inputs['fake_label']
        
        batch_size = real_image_A.size(0)

        # Generate fake images
        (fake_image_B, fake_mask_B, _) = self.netG_A2B(real_image_A)

        output = {
            'real_image_A': real_image_A,
            'real_image_B': real_image_B,
            'fake_image_B': fake_image_B.data,
            'fake_mask_B': fake_mask_B
            }
        return output

    def create_generator(self, input_nc, output_nc, ngf, netG, device, use_dropout=False, init_type='normal', init_gain=0.02):
        """Create a generator
        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ngf (int) -- the number of filters in the last conv layer
            netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
            use_dropout (bool) -- if use dropout layers.
            init_type (str)    -- the name of our initialization method.
            init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        
        Returns a generator
        Our current implementation provides two types of generators:
            U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
            The original U-Net paper: https://arxiv.org/abs/1505.04597
            Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
            Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
            We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
        The generator has been initialized by <init_net>. It uses RELU for non-linearity.
        """
        net = None
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        if netG == 'ResNet9':
            net = ResNetGenerator9(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
        # elif netG == 'resnet_9blocks':
        #     net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
        # elif netG == 'resnet_6blocks':
        #     net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
        # elif netG == 'CustomGenerator':
        #     net = CustomGenerator()
        # elif netG == 'unet_128':
        #     net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # elif netG == 'unet_256':
        #     net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
        return self.init_net(net, device, netG, init_type, init_gain)

    def create_discriminator(self, input_nc, ndf, netD, device, n_layers_D=3, init_type='normal', init_gain=0.02):
        """Create a discriminator
        Parameters:
            input_nc (int)     -- the number of channels in input images
            ndf (int)          -- the number of filters in the first conv layer
            netD (str)         -- the architecture's name: basic | n_layers | pixel
            n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
            init_type (str)    -- the name of the initialization method.
            init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

        Returns a discriminator
        Our current implementation provides three types of discriminators:
            [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
            It can classify whether 70×70 overlapping patches are real or fake.
            Such a patch-level discriminator architecture has fewer parameters
            than a full-image discriminator and can work on arbitrarily-sized images
            in a fully convolutional fashion.
            [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
            with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
            [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
            It encourages greater color diversity but has no effect on spatial statistics.
        The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
        """
        net = None
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        if netD == 'basic':  # default PatchGAN classifier
            net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        elif netD == 'n_layers':  # more options
            net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        elif netD == 'pixel':     # classify if each pixel is real or fake
            net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        elif netD == 'CustomDiscriminator':
            net = CustomDiscriminator()
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
        
        return self.init_net(net, device, netD, init_type, init_gain)    

    # custom weights initialization called on netG and netD
    def init_net(self, net, device, net_name, init_type='normal', init_gain=0.02):
        """Initialize a network: 
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            
        Return an initialized network.
        """
        self.init_weights(net, net_name, init_type, init_gain=init_gain)
        return net.to(device)

    def init_weights(self, net, net_name, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_func)  # apply the initialization function <init_func>

        print(f'Initialized network {net_name} with {init_type} weights.')

    def save_training_images(self, inputs, output, epoch, iter):
        """ Save images """
        # Prepare inputs
        real_image_A= inputs['real_image_A']
        real_image_B = inputs['real_image_B']
        real_label = inputs['real_label']
        fake_label = inputs['fake_label']
        batch_size = real_image_A.size(0)

        # Save A2B
        vutils.save_image(torch.cat((real_image_A, output['fake_image_B']), 0),
                f"{self.args.outf}/{self.args.dataset}/train/A2B/epoch_{epoch}_batch_{iter}.png",
                normalize=True, nrow=batch_size+1,padding=25)
        
        fake_mask_B = torch.nn.Sigmoid()(output['fake_mask_B'])

        vutils.save_image(fake_mask_B, f"{self.args.outf}/{self.args.dataset}/train/A2B/fake_mask_B_epoch_{epoch}_batch_{iter}.png", normalize=True)



        # Save separately each image
        vutils.save_image(real_image_A, f"{self.args.outf}/{self.args.dataset}/train/A2B/real_samples_epoch_{epoch}_batch_{iter}.png", normalize=True)
        vutils.save_image(output['fake_image_B'], f"{self.args.outf}/{self.args.dataset}/train/A2B/fake_samples_epoch_{epoch}_batch_{iter}.png", normalize=True)
    
    def save_test_images(self, output, norm_output, iter):
        """ Save test images """
        A2B_PATH = os.path.join(self.args.outf, self.args.dataset, "test", self.args.input_type, "A2B")
        # B2A_PATH = os.path.join(self.args.outf, self.args.dataset, "test", self.args.input_type, "B2A")

        real_image_A = norm_output['real_image_A']
        real_image_B = norm_output['real_image_B']
        fake_image_B = norm_output['fake_image_B']
        diff_A2B = norm_output['diff_A2B']

        fake_mask_B = torch.nn.Sigmoid()(output['fake_mask_B'].cpu())
                
        plt.figure()

        plt.subplot(1, 4, 1), plt.imshow(real_image_A), plt.axis('off'), plt.title('A', fontsize=8)
        plt.subplot(1, 4, 2), plt.imshow(fake_image_B), plt.axis('off'), plt.title('G(A)', fontsize=8)
        plt.subplot(1, 4, 3), plt.imshow(diff_A2B, cmap='magma'), plt.axis('off'), plt.title('Gray|A - G(A)|', fontsize=8)
        plt.subplot(1, 4, 4), plt.imshow(fake_mask_B.squeeze(0).data, cmap='gray'), plt.axis('off'), plt.title('Fake Mask', fontsize=8)
        plt.savefig(f"{A2B_PATH}/{iter + 1:03d}.png", dpi=300, bbox_inches='tight')
        plt.clf()
        
        plt.imshow(real_image_A), plt.axis('off')
        plt.savefig(f"{A2B_PATH}/real_{iter + 1:03d}.png", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.imshow(fake_image_B), plt.axis('off')
        plt.savefig(f"{A2B_PATH}/fake_{iter + 1:03d}.png", dpi=300, bbox_inches='tight')
        plt.clf()

        plt.close()       

    def mask_threshold(self, mask, threshold):
        """ Apply threshold to raw logits segmented image """
        
        return torch.nn.Sigmoid()(mask) > threshold
        
    def metrics_initialization(self):
        """ Metrics initialization """
        self.metrics = dict()
        self.thrs = np.linspace(0, 1, 10)
        for t in self.thrs:
            # Create dictionary for threshold t
            self.metrics[str(t)] = dict()
            self.metrics[str(t)]["RMSE_back"] = []
            self.metrics[str(t)]["RMSE_def"] = []

    def metrics_evaluation(self, output, mask):
        """ Evaluate metrics for the model """

        real_image_A = output['real_image_A']
        fake_image_B = output['fake_image_B']
        fake_mask_B = mask

        for t in self.thrs:
            # Retrieve mask
            mask = self.mask_threshold(fake_mask_B, t).squeeze(0).cpu().numpy()
            real_mask_def = real_image_A[mask]
            real_mask_back = real_image_A[~mask]
            fake_mask_def = fake_image_B[mask]
            fake_mask_back = fake_image_B[~mask]           
            if len(real_mask_def) > 0:
                self.metrics[str(t)]["RMSE_def"].append(rmse(real_mask_def[np.newaxis, ...], fake_mask_def[np.newaxis, ...], max_p=255))
            if len(real_mask_back) > 0:
                self.metrics[str(t)]["RMSE_back"].append(rmse(real_mask_back[np.newaxis, ...], fake_mask_back[np.newaxis, ...], max_p=255))


    def print_evaluation_metrics(self):
        """ Print evaluation metrics """

        # Compute total metrics
        plt.figure()
        defect_list = []
        background_list = []
        for t in self.thrs:
            l = self.metrics[str(t)]["RMSE_back"]
            if len(l):
                background_list.append(np.asarray(l).mean())
            else:
                background_list.append(np.NaN)

            l = self.metrics[str(t)]["RMSE_def"]
            if len(l):
                defect_list.append(np.asarray(l).mean())
            else:
                defect_list.append(np.NaN)

        plt.plot(self.thrs, np.asarray(background_list), marker='o', linestyle='--', color='r', label='Square', linewidth=0.8)
        plt.ylabel(r"$RMSE$")
        plt.xlabel(r"$t$")
        plt.plot(self.thrs, np.asarray(defect_list), marker='o', linestyle='--', color='b', label='Circle', linewidth=0.8)
        plt.tight_layout()
        plt.xlim(0, 1)
        plt.ylim(min(0.55 * np.nanmin(background_list), .95 * np.nanmin(defect_list)), 
                 max(1.25 * np.nanmax(background_list), 1.25 * np.nanmax(defect_list)))
        plt.legend(['Back', 'Def'])
        plt.savefig('./metrics.png', dpi=300)
        plt.close()

 
    def save_training_progress(self, inputs, mean, std, epoch):
        """ Save training progress for first batch """

        A2B_PATH = os.path.join(self.args.outf, self.args.dataset, "train_progress/A2B")
        
        B_mask_real = inputs['real_image_B_mask']
        output = self.test(inputs)

        fake_mask_B = torch.nn.Sigmoid()(output['fake_mask_B'])

        # Normalization values
        std = torch.tensor(std, device=self.device).view(3, 1, 1)
        mean = torch.tensor(mean, device=self.device).view(3, 1, 1)

        norm_output = normalize_images_diff(output, mean, std)       

        real_image_A = norm_output['real_image_A']
        real_image_B = norm_output['real_image_B']
        fake_image_B = norm_output['fake_image_B']
        diff_A2B = norm_output['diff_A2B']
           
        plt.figure()
        fake_mask_B = (fake_mask_B.squeeze(0).detach().cpu().numpy()*255).astype(np.uint8)
        B_mask_real = (B_mask_real.squeeze(0).detach().cpu().numpy()*255).astype(np.uint8)

        plt.subplot(1, 6, 1), plt.imshow(real_image_A), plt.axis('off'), plt.title('A', fontsize=8)
        plt.subplot(1, 6, 2), plt.imshow(fake_image_B), plt.axis('off'), plt.title('G(A)', fontsize=8)
        plt.subplot(1, 6, 3), plt.imshow(diff_A2B, cmap='magma'), plt.axis('off'), plt.title('Gray|A - G(A)|', fontsize=8)
        plt.subplot(1, 6, 4), plt.imshow(fake_mask_B, cmap='gray'), plt.axis('off'), plt.title('Fake Mask', fontsize=8)
        plt.subplot(1, 6, 5), plt.imshow(B_mask_real, cmap='gray'), plt.axis('off'), plt.title('Real Mask', fontsize=8)
        plt.subplot(1, 6, 6), plt.imshow(real_image_B), plt.axis('off'), plt.title('B', fontsize=8)
        plt.savefig(f"{A2B_PATH}/{epoch:03d}.png", dpi=300, bbox_inches='tight')
        plt.clf()

        plt.close()           