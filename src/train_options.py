from src.base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.It also includes shared options defined in BaseOptions."""
    def initialize(self, parser):
        # Initialize base options
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument("--weightsf", default="./weights", help="folder saving weights. (default:'./weights').")
        parser.add_argument("--netG_A2B", default="", help="path to netG_A2B (to continue training)")
        parser.add_argument("--netG_B2A", default="", help="path to netG_B2A (to continue training)")
        parser.add_argument("--netD_A", default="", help="path to netD_A (to continue training)")
        parser.add_argument("--netD_B", default="", help="path to netD_B (to continue training)")
        parser.add_argument("--netD_fit", default="", help="path to netD_fit (to continue training)")
        parser.add_argument("--netD_mask", default="", help="path to netD_mask (to continue training)")
        # training parameters
        parser.add_argument("--decay_epochs", type=int, default=100, help="epoch to start linearly decaying the learning rate to 0. (default:100)")
        parser.add_argument("--batch_size", default=1, type=int, metavar="N", help="mini-batch size (default: 1), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel")
        parser.add_argument("--lr", type=float, default=0.0002, help="learning rate. (default:0.0002)")
        parser.add_argument("--beta_1", type=float, default=0.5, help="Beta 1 optimizer. (default:0.5)")
        parser.add_argument("--beta_2", type=float, default=0.999, help="Beta 2 optimizer. (default:0.999)")    
        parser.add_argument("--lambda_identity_A", type=float, default=5.0, help="Weight identity A loss")    
        parser.add_argument("--lambda_identity_B", type=float, default=5.0, help="Weight identity B loss")    
        parser.add_argument("--lambda_GAN_A2B", type=float, default=1.0, help="Weight adversarial A2B loss")    
        parser.add_argument("--lambda_GAN_B2A", type=float, default=1.0, help="Weight adversarial B2A loss")    
        parser.add_argument("--lambda_cycle_ABA", type=float, default=10.0, help="Weight cycle B2A loss")    
        parser.add_argument("--lambda_cycle_BAB", type=float, default=10.0, help="Weight cycle B2A loss")  
        parser.add_argument("--lambda_GAN_fit", type=float, default=150.0, help="Weight adversarial fit loss")  
        parser.add_argument("--lambda_background", type=float, default=0.1, help="Weight background loss")  

        return parser
