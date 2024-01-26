import torch
import torch.nn as nn
from torch.autograd import Variable
from pdb import set_trace as st
import torch.nn.functional as F
import torch.optim as optim

###########################
# Autoencoder
###########################
# You can modify the model using convolutional layer


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or  classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class AutoEncoder(nn.Module):
    def __init__(self, input_nA, input_nB, input_nC, output_nC):
        super(AutoEncoder, self).__init__()
        # input_A = 1750 input_B = 3073 input_C = 19303 output_C = 39375
        self.input_nA = input_nA
        self.input_nB = input_nB
        self.input_nC = input_nC
        self.output_nC = output_nC
        self.layer1_1 = nn.Linear(input_nA, 2048, bias = False)
        self.layer1_2 = nn.Linear(input_nB, 2048, bias = False)
        self.layer1_3 = nn.Linear(input_nC, 2048, bias = False)
        self.layer2 = nn.Linear(2048, 1024)
        self.layer3 = nn.Linear(1024, 256)
        self.layer4 = nn.Linear(256, 1024)
        self.layer5 = nn.Linear(1024, 2048)
        self.layer6_1 = nn.Linear(2048, input_nA)
        self.layer6_2 = nn.Linear(2048, input_nB)
        self.layer6_3 = nn.Linear(2048, input_nC)
        self.layer6_4 = nn.Linear(2048, output_nC)
        self.drop = 0.5
        self.beta1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.beta2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward_ac2b(self, x1, x3):
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, self.input_nA))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, self.input_nC))), self.drop)

        x = (x1 + x3) / 2
        x = F.dropout(F.relu(self.layer2(x)), self.drop)
        self.com1 = F.dropout(F.relu(self.layer3(x)), self.drop)

        x = F.dropout(F.relu(self.layer4(self.com1)), self.drop)
        x = F.dropout(F.relu(self.layer5(x)), self.drop)

        out1 = F.relu(self.layer6_1(x))
        out2 = torch.tanh(self.layer6_2(x))
        out3 = torch.sigmoid(self.layer6_3(x))

        out1 = out1.view(1, 1, self.input_nA)
        out2 = out2.view(1, 1, self.input_nB)
        out3 = out3.view(1, 1, self.input_nC)
        return out1, out2, out3, self.com1

    def forward_bc2a(self, x2, x3):
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, self.input_nB))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, self.input_nC))), self.drop)

        x = (x2+x3)/2
        x = F.dropout(F.relu(self.layer2(x)), self.drop)

        self.com2 = F.dropout(F.relu(self.layer3(x)), self.drop)

        x = F.dropout(F.relu(self.layer4(self.com2)), self.drop)
        x = F.dropout(F.relu(self.layer5(x)), self.drop)

        out1 = F.relu(self.layer6_1(x))
        out2 = torch.tanh(self.layer6_2(x))
        out3 = torch.sigmoid(self.layer6_3(x))

        out1 = out1.view(1,1,self.input_nA)
        out2 = out2.view(1,1,self.input_nB)
        out3 = out3.view(1,1,self.input_nC)
        return out1, out2, out3, self.com2

    def forward_ab2c(self, x1, x2):
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, self.input_nA))), self.drop)
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, self.input_nB))), self.drop)

        x = (x2+x1)/2
        x = F.dropout(F.relu(self.layer2(x)), self.drop)

        self.com3 = F.dropout(F.relu(self.layer3(x)), self.drop)

        x = F.dropout(F.relu(self.layer4(self.com3)), self.drop)
        x = F.dropout(F.relu(self.layer5(x)), self.drop)

        out1 = F.relu(self.layer6_1(x))
        out2 = torch.tanh(self.layer6_2(x))
        out3 = torch.sigmoid(self.layer6_4(x))

        out1 = out1.view(1,1,self.input_nA)
        out2 = out2.view(1,1,self.input_nB)
        out3 = out3.view(1,1,self.output_nC)
        return out1, out2, out3, self.com3

    def forward_abc2c(self, x1, x2, x3):
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, self.input_nA))), self.drop)
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, self.input_nB))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, self.input_nC))), self.drop)

        x = (x3+x2+x1)/3
        x = F.dropout(F.relu(self.layer2(x)), self.drop)
        self.com3 = F.dropout(F.relu(self.layer3(x)), self.drop)

        x = F.dropout(F.relu(self.layer4(self.com3)), self.drop)
        x = F.dropout(F.relu(self.layer5(x)), self.drop)

        out1 = F.relu(self.layer6_1(x))
        out2 = torch.tanh(self.layer6_2(x))
        out3 = torch.sigmoid(self.layer6_3(x))
        out4 = torch.sigmoid(self.layer6_4(x))

        out1 = out1.view(1,1,self.input_nA)
        out2 = out2.view(1,1,self.input_nB)
        out3 = out3.view(1,1,self.input_nC)
        out4 = out4.view(1,1,self.output_nC)

        return out1, out2, out3, out4, self.com3


def define_AE(input_A, input_B, input_C, output_C, device):

    NetAE = AutoEncoder(input_A, input_B, input_C, output_C)
    NetAE.to(device)
    NetAE.apply(weights_init)

    return NetAE


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nD, ndf, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.input_nD = input_nD
        # self.drpout = 0.5
        kw = 4
        sequence = [
            nn.Linear(self.input_nD,ndf),
            nn.LeakyReLU(0.2, True),
            # nn.Dropout(self.dropout),
            nn.Linear(ndf,1),
        ]
        sequence += [nn.Sigmoid()]
        #说是不用sigmoid结果还是用了sigmoid
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor):
            #return nn.parallel.data_parallel(self.model, input.view(1,1,28,28), self.gpu_ids)
            #print(input)
            return self.model(input.view(self.input_nD))
        else:
            return self.model(input.view(self.input_nD))



def define_D(input_nc, ndf, which_model_netD, device, n_layers_D=3, use_sigmoid=False):

    if which_model_netD == 'basic':
        netD = define_D(input_nc, ndf, 'n_layers', device, use_sigmoid = use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid)
    else:
        print('Discriminator model name [%s] is not recognized' %which_model_netD)

    netD.to(device)
    netD.apply(xavier_init)

    return netD



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)

        return x

def define_E(in_dim, hgcn_dim, device,dropout=0.5):

    NetE = GCN_E(in_dim, hgcn_dim, dropout)
    NetE.to(device)

    return NetE


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

def define_C(in_dim, out_dim, device):

    NetC = Classifier_1(in_dim, out_dim)
    NetC.to(device)

    return NetC



class VCDN(nn.Module):
    def __init__(self, num_feature, num_label, hvcdn_dim):
        super().__init__()
        self.num_label = num_label
        self.model = nn.Sequential(
            nn.Linear(pow(num_label, num_feature), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_label)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
                          (-1, pow(self.num_label, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_label, i + 1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_label, num_view)))
        output = self.model(vcdn_feat)
        prob = torch.sigmoid(output)
        return output, prob

def define_VCDN(num_feature, num_label, hvcdn_dim, device):

    NetVCDN = VCDN(num_feature, num_label, hvcdn_dim)
    NetVCDN.to(device)

    return NetVCDN


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i + 1)].parameters()) + list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict
###############################################################################
# Functions
###############################################################################




def define_G(input_nc, output_nc, ngf, which_model_netG, norm, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = InstanceNormalization
    else:
        print('normalization layer [%s] is not found' % norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
        #netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG





def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss used in LSGAN.
# It is basically same as MSELoss, but it abstracts away the need to create
# the target label tensor that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 # target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, device = []):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.device = device
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = torch.tensor(input.size(), dtype=torch.float, requires_grad=False)\
                    .fill_(self.real_label).to(self.device)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = torch.tensor(input.size(), dtype=torch.float, requires_grad=False) \
                    .fill_(self.fake_label).to(self.device)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc #input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        #####################################################
        ####WQQQQQQ#####
        model = [nn.Linear(self.input_nc,64),
                 nn.ReLU(True),
                 nn.Linear(64,self.output_nc),
                 nn.ReLU()]
        #nn.ReLU()   nn.Tanh()
#        if input_nc<100:
#            model = [nn.Linear(self.input_nc,64),
#                 nn.ReLU(True),
#                 nn.Linear(64,self.output_nc),
#                 nn.Tanh()]
#        else:
#            model = [nn.Linear(self.input_nc,600),
#                 nn.ReLU(True),
#                 nn.Linear(600,self.output_nc),
#                 nn.Tanh()]

#        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
#                 norm_layer(ngf),
#                 nn.ReLU(True)]
#        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
#                 norm_layer(ngf),
#                 nn.ReLU(True)]
#
#        n_downsampling = 2
#        for i in range(n_downsampling):
#            mult = 2**i
#            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                                stride=2, padding=1),
#                      norm_layer(ngf * mult * 2),
#                      nn.ReLU(True)]
#
#        mult = 2**n_downsampling
#        for i in range(n_blocks):
#            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer)]
#
#        for i in range(n_downsampling):
#            mult = 2**(n_downsampling - i)
#            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                         kernel_size=3, stride=2,
#                                         padding=1, output_padding=1),
#                      norm_layer(int(ngf * mult / 2)),
#                      nn.ReLU(True)]
#
#        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]

        #self.model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            #return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            #print(input.view(784))
            return self.model(input.view(self.input_nc))
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer)

    def build_conv_block(self, dim, padding_type, norm_layer):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True)

        self.model = unet_block

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)




# Instance Normalization layer from
# https://github.com/darkstar112358/fast-neural-style

class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
