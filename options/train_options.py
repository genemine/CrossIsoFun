from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--dim_hvcdn', type =int, default = 512, help='dimension of vcdn interlayer')
        self.parser.add_argument('--adj_parameter', type =int, default = 5, help='average # of edges retained per node, a')
        self.parser.add_argument('--GCN1', type =int, default = 400, help='dimension of the 1st convolutional layer of GCN')
        self.parser.add_argument('--GCN2', type =int, default = 400, help='dimension of the 2nd convolutional layer of GCN')
        self.parser.add_argument('--GCN3', type =int, default = 200, help='dimension of the 3rd convolutional layer of GCN')
        self.parser.add_argument('--num_GO', type =int, default = 96, help='# of GO terms input')
        self.parser.add_argument('--continue_train_after_GCN', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--continue_train_after_AE', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=15, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=15, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--ntrain', type=int, default=float("inf"), help='# of examples per epoch.')
        self.parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_C', type=float, default=1.0, help='weight for AE(Gx) loss in VIGAN')  #1.0
        self.parser.add_argument('--lambda_D', type=float, default=10.0, help='weight for AE loss in VIGAN') #10.0
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # NOT-IMPLEMENTED self.parser.add_argument('--preprocessing', type=str, default='resize_and_crop', help='resizing/cropping strategy')
        self.isTrain = True
