import numpy as np
import torch
import os
from collections import OrderedDict
from pdb import set_trace as st
import itertools
import util.util as util
from util.image_pool import ImagePool
from base_model import BaseModel
import networks as networks
import sys
from generate_adj_matrix import gen_tr_adj_mat, gen_te_adj_mat
from cal_auc import *
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_scatter import scatter_max


class VIGANModel(BaseModel):
    def name(self):
        return 'VIGANModel'

    def initialize(self, opt, num_feature, num_label, dim_list, dim_he_list, dim_hvcdn, device,
                   lr_e_pretrain_1=1e-3,
                   lr_e_pretrain_2=1e-4,
                   lr_e_pretrain_3=1e-3,
                   lr_e_1=1e-4,
                   lr_e_2=2e-4,
                   lr_e_3=5e-4,
                   lr_c=1e-5
                   ):

        BaseModel.initialize(self, opt)

        self.gene_PPI_column = dim_list[2]
        self.device = device

        # load/define networks
        ###################################
        self.device = device

        self.E1 = networks.define_E(dim_list[0], dim_he_list, self.device, dropout=0.4)
        self.C1 = networks.define_C(dim_he_list[-1], num_label,self.device)
        self.E2 = networks.define_E(dim_list[1], dim_he_list, self.device, dropout=0.2)
        self.C2 = networks.define_C(dim_he_list[-1], num_label, self.device)
        self.E3 = networks.define_E(dim_list[3], dim_he_list, self.device, dropout=0.8)
        self.C3 = networks.define_C(dim_he_list[-1], num_label, self.device)

        self.VCDN = networks.define_VCDN(num_feature, num_label, dim_hvcdn, self.device)
        #####################################
        if self.isTrain:
            self.AE = networks.define_AE(dim_list[0], dim_list[1], dim_list[2], dim_list[3], self.device)
            self.netD_C = networks.define_D(dim_list[2], opt.ndf_C, opt.which_model_netD, self.device)

        if opt.continue_train_after_AE:
            # which_epoch = 'after_AE_pretrain'
            which_epoch = ' '
            self.load_network(self.AE, 'AE', which_epoch)
            self.load_network(self.netD_C, 'D_C', which_epoch)

        if opt.continue_train_after_GCN:
            which_epoch = ' '
            self.load_network(self.C1, 'C1', which_epoch)
            self.load_network(self.E1, 'E1', which_epoch)
            self.load_network(self.C2, 'C2', which_epoch)
            self.load_network(self.E2, 'E2', which_epoch)
            self.load_network(self.C3, 'C3', which_epoch)
            self.load_network(self.E3, 'E3', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_C_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, gpu_ids=self.gpu_ids)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionCycle_C = torch.nn.BCELoss()
            self.criterionAE = torch.nn.MSELoss()
            self.criterionAE_C = torch.nn.BCELoss()
            self.criterionCycle_C = torch.nn.L1Loss()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.criterionVCDN = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.criterionGCN = torch.nn.BCEWithLogitsLoss(reduction='none')

            # initialize optimizers
            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.AE.parameters(),
                                                lr=3e-5, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(),
                                                  lr=1e-8, betas=(opt.beta1, 0.999), weight_decay=1e-4)

            self.optimizer_E1_C1_pretrain = torch.optim.Adam(
                itertools.chain(self.E1.parameters(), self.C1.parameters()), lr=lr_e_pretrain_1)  # lr_e_pretrain=5e-3
            self.optimizer_E2_C2_pretrain = torch.optim.Adam(
                itertools.chain(self.E2.parameters(), self.C2.parameters()), lr=lr_e_pretrain_2)  # 1e-4 原始是5e-4
            # self.optimizer_E3_C3_pretrain = torch.optim.Adam(itertools.chain(self.E3.parameters(), self.C3.parameters()), lr=1e-4)
            self.optimizer_AE_GCN_C = torch.optim.Adam(itertools.chain(  # self.AE.parameters(),
                # self.netD_A.parameters(),
                # self.netD_B.parameters(),
                # self.netD_C.parameters(),
                self.E3.parameters(), self.C3.parameters()), lr=lr_e_pretrain_3)  # 1e-4

            # self.optimizer_AE_GCN = torch.optim.Adam(itertools.chain(self.AE.parameters(), \
            #                                                          # self.netD_A.parameters(),
            #                                                          # self.netD_B.parameters(),
            #                                                          # self.netD_C.parameters(),
            #                                                          self.E1.parameters(), self.C1.parameters(), \
            #                                                          self.E2.parameters(), self.C2.parameters(), \
            #                                                          self.E3.parameters(), self.C3.parameters()),
            #                                          lr=lr_e_pretrain)

            self.optimizer_E1_C1 = torch.optim.Adam(itertools.chain(self.E1.parameters(), self.C1.parameters()),
                                                    lr=lr_e_1)
            self.optimizer_E2_C2 = torch.optim.Adam(itertools.chain(self.E2.parameters(), self.C2.parameters()),
                                                    lr=lr_e_2)
            self.optimizer_E3_C3 = torch.optim.Adam(itertools.chain(self.E3.parameters(), self.C3.parameters()),
                                                    lr=lr_e_3)

            self.optimizer_GCN_VCDN = torch.optim.Adam(itertools.chain(
                self.VCDN.parameters()), lr=lr_c)

            print('---------- Networks initialized -------------')
            networks.print_network(self.AE)
            # networks.print_network(self.netD_A)
            # networks.print_network(self.netD_B)
            networks.print_network(self.netD_C)
            networks.print_network(self.E1)
            networks.print_network(self.E2)
            networks.print_network(self.E3)
            networks.print_network(self.C1)
            networks.print_network(self.C2)
            networks.print_network(self.C3)
            networks.print_network(self.VCDN)
            print('-----------------------------------------------')

    # def set_input(self, data_a, data_b, data_c):
    #
    #     self.input_A = data_a#.cuda(self.gpu_ids[0])
    #     self.input_B = data_b#.cuda(self.gpu_ids[0])
    #     self.input_C = data_c#.cuda(self.gpu_ids[0])

    def set_input(self, data_a, data_b, data_c, iso2gene):

        self.input_A = data_a.to(self.device)
        self.input_B = data_b.to(self.device)
        self.input_C = data_c.to(self.device)
        self.iso2gene = iso2gene.to(self.device)

    def forward(self):

        self.real_A = self.input_A
        self.real_B = self.input_B
        self.real_C = self.input_C


    def isoform2gene_PPI(self, isoform_PPI):
        gene_PPI, _ = scatter_max(isoform_PPI, self.iso2gene, dim=-1)
        return gene_PPI


    def calculate_class_weights(self, target):
        class_counts = torch.bincount(target)
        total_samples = len(target)
        self.class_weights = total_samples / len(class_counts) / class_counts.float()
        print(self.class_weights)


    def class_weights_for_PPI_input(self, target):
        try:
            target = torch.where(target == 0, self.class_weights[0], target)
            target = torch.where(target == 1, self.class_weights[1], target)
        except:
            target = torch.ones(len(target)) * self.class_weights[0].to(self.device)
        return target


    def set_input_AE_GCN(self, data_list, adj_list, labels_tensor, sample_weight, iso_gene):

        self.data_list = data_list
        self.adj_list = adj_list

        self.labels = labels_tensor.to(self.device)
        self.labels_weight = sample_weight.to(self.device)
        self.iso_gene_paired = iso_gene[0].to(self.device)
        self.iso_gene_unpaired = iso_gene[1].to(self.device)

        self.num_gene = len(labels_tensor)


    def forward_AE_GCN(self):

        self.A_matrix = self.data_list[0].to(self.device)
        self.B_matrix = self.data_list[1].to(self.device)
        self.C_matrix = self.data_list[2].to(self.device)

        self.A_adj = self.adj_list[0].to(self.device)
        self.B_adj = self.adj_list[1].to(self.device)
        self.C_adj = self.adj_list[2].to(self.device)


    def set_input_AE_GCN_test(self, data_list, adj_list, trte_idx, iso_gene):

        self.data_list = data_list
        self.adj_list = adj_list

        self.iso_gene_paired = iso_gene[0].to(self.device)
        self.iso_gene_unpaired = iso_gene[1].to(self.device)


    def forward_AE_GCN_test(self):

        self.A_matrix = self.data_list[0].to(self.device)
        self.B_matrix = self.data_list[1].to(self.device)
        self.C_matrix = self.data_list[2].to(self.device)

        self.A_adj = self.adj_list[0].to(self.device)
        self.B_adj = self.adj_list[1].to(self.device)
        self.C_adj = self.adj_list[2].to(self.device)

    def set_input_GCN_VCDN(self, data_list, adj_list, labels_tensor, sample_weight, iso_gene):

        self.data_list = data_list
        self.adj_list = adj_list

        self.labels = labels_tensor.to(self.device)
        self.labels_weight = sample_weight.to(self.device)
        self.iso_gene_paired = iso_gene[0].to(self.device)
        self.iso_gene_unpaired = iso_gene[1].to(self.device)

    def forward_GCN_VCDN(self):

        self.A_matrix = self.data_list[0].to(self.device)  # .cuda(self.gpu_ids[0])
        self.B_matrix = self.data_list[1].to(self.device)  # .cuda(self.gpu_ids[0])
        self.C_matrix = self.data_list[2].to(self.device)  # .cuda(self.gpu_ids[0])

        self.A_adj = self.adj_list[0].to(self.device)
        self.B_adj = self.adj_list[1].to(self.device)
        self.C_adj = self.adj_list[2].to(self.device)

    def test_unpaired(self):
        with torch.no_grad():
            self.real_A = self.input_A
            self.real_B = self.input_B
            self.real_C = self.input_C

        AErealA1, self.fake_B, AErealC1, self.com1 = self.AE.forward_ac2b(self.real_A, self.real_C)

        self.fake_A, AErealB2, AErealC2, self.com2 = self.AE.forward_bc2a(self.real_B, self.real_C)

        AErealA3, AErealB3, self.fake_C, self.com3 = self.AE.forward_ab2c(self.real_A, self.real_B)

        return self.fake_BA, self.fake_AB, self.com1, self.com2

    def test_unpaired_a(self):
        with torch.no_grad():
            self.real_B = self.input_B
            self.real_C = self.input_C

        self.fake_A, AErealB2, AErealC2, self.com2 = self.AE.forward_bc2a(self.real_B, self.real_C)

        return self.fake_A

    def test_unpaired_b(self):
        with torch.no_grad():
            self.real_A = self.input_A
            self.real_C = self.input_C

        AErealA1, self.fake_B, AErealC1, self.com1 = self.AE.forward_ac2b(self.real_A, self.real_C)

        return self.fake_B

    def test_unpaired_c(self):
        with torch.no_grad():
            self.real_A = self.input_A
            self.real_B = self.input_B
            self.real_C = self.input_C

        AErealA4, AErealB4, AErealC4, self.fake_D, _ = self.AE.forward_abc2c(self.real_A, self.real_B, self.real_C)

        return self.fake_D.detach()


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()

        return loss_D

    def backward_D_unpaired(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        # loss_D.backward()

        return loss_D, loss_D_fake

    def backward_D_C(self):
        fake_C = self.fake_C_pool.query(self.fake_D)
        self.loss_D_C = self.backward_D_basic(self.netD_C, self.real_C, fake_C)

    def backward_D_C_unpaired(self):
        fake_C = self.fake_C_pool.query(self.fake_D)
        self.loss_D_C, self.loss_D_C_f = self.backward_D_unpaired(self.netD_C, self.real_C, fake_C)


    ############################################################################
    # Define backward function for VIGAN
    ############################################################################
    def backward_AE_pretrain(self):
        # Autoencoder loss

        self.fake_A, AErealB1, AErealC1, _ = self.AE.forward_bc2a(self.real_B, self.real_C)

        AErealA2, self.fake_B, AErealC2, _ = self.AE.forward_ac2b(self.real_A, self.real_C)

        AErealA3, AErealB3, self.fake_C, _ = self.AE.forward_ab2c(self.real_A, self.real_B)

        AErealA4, AErealB4, self.recover_C, _ = self.AE.forward_abc2c(self.real_A, self.real_B, self.real_C)

        self.loss_AE_A = self.criterionAE(self.fake_A, self.real_A) + \
                         self.criterionAE(AErealB1, self.real_B) + \
                         self.criterionAE(AErealC1, self.real_C)

        self.loss_PPI_A = self.criterionAE(self.fake_A, self.real_A)

        # self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + \
        #                  self.criterionAE(self.fake_B, self.real_B) + \
        #                  self.criterionAE(AErealC2, self.real_C)

        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + \
                         self.criterionAE(AErealC2, self.real_C)

        # self.loss_PPI_B = self.criterionAE(self.fake_B, self.real_B)

        self.loss_AE_C = self.criterionAE(self.fake_C, self.real_C) + \
                         self.criterionAE(AErealB3, self.real_B) + \
                         self.criterionAE(AErealA3, self.real_A)

        self.loss_PPI_C = self.criterionAE(self.fake_C, self.real_C)

        self.loss_AE_D = self.criterionAE(self.recover_C, self.real_C) + \
                         self.criterionAE(AErealB4, self.real_B) + \
                         self.criterionAE(AErealA4, self.real_A)

        self.loss_PPI_D = self.criterionAE(self.recover_C, self.real_C)

        self.loss_AE_pre = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C + self.loss_AE_D

        self.loss_AE_pre.backward()

    def test_AE_pretrain(self):
        # Autoencoder loss

        self.fake_A, AErealB1, AErealC1, self.com1 = self.AE.forward_bc2a(self.real_B, self.real_C)

        AErealA2, self.fake_B, AErealC2, self.com2 = self.AE.forward_ac2b(self.real_A, self.real_C)

        AErealA3, AErealB3, self.fake_C, self.com3 = self.AE.forward_ab2c(self.real_A, self.real_B)

        AErealA4, AErealB4, self.recover_C, _ = self.AE.forward_abc2c(self.real_A, self.real_B, self.real_C)

        self.loss_AE_A = self.criterionAE(self.fake_A, self.real_A) + \
                         self.criterionAE(AErealB1, self.real_B) + \
                         self.criterionAE(AErealC1, self.real_C)

        self.loss_PPI_A = self.criterionAE(self.fake_A, self.real_A)

        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + \
                         self.criterionAE(AErealC2, self.real_C)

        # self.loss_PPI_B = self.criterionAE(self.fake_B, self.real_B)

        self.loss_AE_C = self.criterionAE(AErealA3, self.real_A) + \
                         self.criterionAE(AErealB3, self.real_B) + \
                         self.criterionAE(self.fake_C, self.real_C)

        self.loss_PPI_C = self.criterionAE(self.fake_C, self.real_C)

        self.loss_AE_D = self.criterionAE(self.recover_C, self.real_C) + \
                         self.criterionAE(AErealB4, self.real_B) + \
                         self.criterionAE(AErealA4, self.real_A)

        self.loss_PPI_D = self.criterionAE(self.recover_C, self.real_C)

        self.loss_AE_pre = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C + self.loss_AE_D

    ########################################################################################################

    def backward_G(self):

        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C

        self.fake_A, AErealB1, AErealC1, _ = self.AE.forward_bc2a(self.real_B, self.real_C)
        AErealA2, self.fake_B, AErealC2, _ = self.AE.forward_ac2b(self.real_A, self.real_C)
        AErealA3, AErealB3, self.fake_C, _ = self.AE.forward_ab2c(self.real_A, self.real_B)
        AErealA4, AErealB4, AErealC4, self.fake_D, _ = self.AE.forward_abc2c(self.real_A, self.real_B, self.real_C)

        self.fake_C = self.isoform2gene_PPI(self.fake_C)
        self.fake_D = self.isoform2gene_PPI(self.fake_D)

        # GAN loss (generator)
        pred_fake = self.netD_C.forward(self.fake_D)
        self.loss_G_C = self.criterionGAN(pred_fake, True)

        self.loss_GABC = self.loss_G_C


        class_weights = self.class_weights_for_PPI_input(self.real_C.flatten())
        self.criterionAE_C = torch.nn.BCELoss(weight=class_weights)
        # self.criterionCycle_C = torch.nn.BCELoss(weight=self.class_weights)

        self.loss_AE_A = self.criterionAE(self.fake_A, self.real_A) + \
                         self.criterionAE(AErealB1, self.real_B) + \
                         self.criterionAE_C(AErealC1, self.real_C)

        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + \
                         self.criterionAE(self.fake_B, self.real_B) + \
                         self.criterionAE_C(AErealC2, self.real_C)

        self.loss_AE_C = self.criterionAE(AErealA3, self.real_A) + \
                         self.criterionAE(AErealB3, self.real_B) + \
                         self.criterionAE_C(self.fake_C, self.real_C)

        self.loss_AE_D = self.criterionAE(AErealA4, self.real_A) + \
                         self.criterionAE(AErealB4, self.real_B) + \
                         self.criterionAE_C(AErealC4, self.real_C) + \
                         self.criterionAE_C(self.fake_D, self.real_C)

        self.loss_AE = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C + self.loss_AE_D

        # Forward cycle loss
        # self.rec_A1, self.rec_B1, self.rec_C1, _ = self.AE.forward_bc2a(self.real_B, self.fake_D)
        self.rec_A1, self.rec_B1, self.rec_C1, _ = self.AE.forward_bc2a(self.fake_B, self.fake_D)
        self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A) * lambda_A
        self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B) * lambda_B
        self.loss_cycle_C1 = self.criterionCycle_C(self.rec_C1, self.real_C) * lambda_C
        self.loss_cyc1 = self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_cycle_C1

        self.rec_A2, self.rec_B2, self.rec_C2, _ = self.AE.forward_ac2b(self.fake_A, self.fake_D)
        self.loss_cycle_A2 = self.criterionCycle(self.rec_A2, self.real_A) * lambda_A
        self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B) * lambda_B
        self.loss_cycle_C2 = self.criterionCycle_C(self.rec_C2, self.real_C) * lambda_C
        self.loss_cyc2 = self.loss_cycle_A2 + self.loss_cycle_B2 + self.loss_cycle_C2
        # self.loss_cyc2 = self.loss_cycle_A2 + self.loss_cycle_C2

        # self.rec_A3, self.rec_B3, self.rec_C3, _ = self.AE.forward_ab2c(self.fake_A, self.real_B)
        self.rec_A3, self.rec_B3, self.rec_C3, _ = self.AE.forward_ab2c(self.fake_A, self.fake_B)
        self.rec_C3 = self.isoform2gene_PPI(self.rec_C3)
        self.loss_cycle_A3 = self.criterionCycle(self.rec_A3, self.real_A) * lambda_A
        self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_B) * lambda_B
        self.loss_cycle_C3 = self.criterionCycle_C(self.rec_C3, self.real_C) * lambda_C
        self.loss_cyc3 = self.loss_cycle_A3 + self.loss_cycle_B3 + self.loss_cycle_C3

        # self.rec_A4, self.rec_B4, self.rec_C4, self.rec_D4, _ = self.AE.forward_abc2c(self.fake_A, self.real_B, self.fake_D)
        self.rec_A4, self.rec_B4, self.rec_C4, self.rec_D4, _ = self.AE.forward_abc2c(self.fake_A, self.fake_B,
                                                                                      self.fake_D)
        self.rec_D4 = self.isoform2gene_PPI(self.rec_D4)
        self.loss_cycle_A4 = self.criterionCycle(self.rec_A4, self.real_A) * lambda_A
        self.loss_cycle_B4 = self.criterionCycle(self.rec_B4, self.real_B) * lambda_B
        self.loss_cycle_C4 = self.criterionCycle_C(self.rec_C4, self.real_C) * lambda_C
        self.loss_cycle_D4 = self.criterionCycle_C(self.rec_D4, self.real_C) * lambda_C
        self.loss_cyc4 = self.loss_cycle_A4 + self.loss_cycle_B4 + self.loss_cycle_C4 + self.loss_cycle_D4

        self.loss_cycle = self.loss_cyc1 + self.loss_cyc2 + self.loss_cyc3 + self.loss_cyc4

        # Generation loss
        self.loss_PPI_A = self.criterionAE(self.fake_A, self.real_A)
        self.loss_PPI_B = self.criterionAE(self.fake_B, self.real_B)
        self.loss_PPI_fake = self.criterionAE_C(self.fake_C, self.real_C)
        self.loss_PPI = self.criterionAE_C(self.fake_D, self.real_C)


        self.loss_G = self.loss_GABC \
                      + self.loss_AE \
                      + 0.05 * self.loss_cycle  # +\

        self.loss_G.backward()

    def backward_G_unpaired(self):

        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        AErealA3, AErealB3, self.fake_C, _ = self.AE.forward_ab2c(self.real_A, self.real_B)
        AErealA4, AErealB4, AErealC4, self.fake_D, _ = self.AE.forward_abc2c(self.real_A, self.real_B, self.real_C)
        self.fake_C = self.isoform2gene_PPI(self.fake_C)
        self.fake_D = self.isoform2gene_PPI(self.fake_D)
        self.fake_A, AErealB1, AErealC1, _ = self.AE.forward_bc2a(self.real_B, self.fake_D)
        AErealA2, self.fake_B, AErealC2, _ = self.AE.forward_ac2b(self.real_A, self.fake_D)

        # GAN loss (Generator)
        pred_fake = self.netD_C.forward(self.fake_D)
        self.loss_G_C = self.criterionGAN(pred_fake, True)

        self.loss_GABC = self.loss_G_C

        # AE loss
        # self.loss_AE_A = self.criterionAE(AErealB1, self.real_B) #+ self.criterionAE_C(AErealC1, self.real_C)
        # self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) #+ self.criterionAE_C(AErealC2, self.real_C)
        # self.loss_AE_C = self.criterionAE(AErealA3, self.real_A) + self.criterionAE(AErealB3, self.real_B)
        # self.loss_AE_D = self.criterionAE(AErealA4, self.real_A) + self.criterionAE(AErealB4, self.real_B) #+ self.criterionAE(AErealC4, self.real_C)

        self.loss_AE_A = self.criterionAE(self.fake_A, self.real_A) + \
                         self.criterionAE(AErealB1, self.real_B)  # + \
        # self.criterionAE_C(AErealC1, self.real_C)

        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + \
                         self.criterionAE(self.fake_B, self.real_B)  # + \
        # self.criterionAE_C(AErealC2, self.real_C)

        self.loss_AE_C = self.criterionAE(AErealA3, self.real_A) + \
                         self.criterionAE(AErealB3, self.real_B)  # + \
        # self.criterionAE_C(self.fake_C, self.real_C)

        self.loss_AE_D = self.criterionAE(AErealA4, self.real_A) + \
                         self.criterionAE(AErealB4, self.real_B)  # + \
        # self.criterionAE_C(AErealC4, self.real_C) + \
        # self.criterionAE_C(self.fake_D, self.real_C)

        self.AErealC1 = AErealC1
        self.AErealC2 = AErealC2
        self.AErealC4 = AErealC4

        self.loss_AE_temp = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C + self.loss_AE_D

        # Forward cycle loss
        # self.rec_A1, self.rec_B1, self.rec_C1, _ = self.AE.forward_bc2a(self.real_B, self.fake_D)
        self.rec_A1, self.rec_B1, self.rec_C1, _ = self.AE.forward_bc2a(self.fake_B, self.fake_D)
        self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A) * lambda_A
        self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B) * lambda_B
        # self.loss_cycle_C1 = self.criterionCycle(self.rec_C1, self.real_C) * lambda_C
        self.loss_cyc1 = self.loss_cycle_A1 + self.loss_cycle_B1  # + self.loss_cycle_C1

        self.rec_A2, self.rec_B2, self.rec_C2, _ = self.AE.forward_ac2b(self.fake_A, self.fake_D)
        self.loss_cycle_A2 = self.criterionCycle(self.rec_A2, self.real_A) * lambda_A
        self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B) * lambda_B
        # self.loss_cycle_C2 = self.criterionCycle(self.rec_C2, self.real_C) * lambda_C
        self.loss_cyc2 = self.loss_cycle_A2 + self.loss_cycle_B2  # + self.loss_cycle_C2
        # self.loss_cyc2 = self.loss_cycle_A2 #+ self.loss_cycle_B2 #+ self.loss_cycle_C2

        # self.rec_A3, self.rec_B3, self.rec_C3, _ = self.AE.forward_ab2c(self.fake_A, self.real_B)
        self.rec_A3, self.rec_B3, self.rec_C3, _ = self.AE.forward_ab2c(self.fake_A, self.fake_B)
        self.rec_C3 = self.isoform2gene_PPI(self.rec_C3)
        self.loss_cycle_A3 = self.criterionCycle(self.rec_A3, self.real_A) * lambda_A
        self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_B) * lambda_B
        # self.loss_cycle_C3 = self.criterionCycle(self.rec_C3, self.real_C) * lambda_C
        self.loss_cyc3 = self.loss_cycle_A3 + self.loss_cycle_B3  # + self.loss_cycle_C3

        # self.rec_A4, self.rec_B4, self.rec_C4, self.rec_D4, _ = self.AE.forward_abc2c(self.fake_A, self.real_B, self.fake_D)
        self.rec_A4, self.rec_B4, self.rec_C4, self.rec_D4, _ = self.AE.forward_abc2c(self.fake_A, self.fake_B,
                                                                                      self.fake_D)
        self.rec_D4 = self.isoform2gene_PPI(self.rec_D4)
        self.loss_cycle_A4 = self.criterionCycle(self.rec_A4, self.real_A) * lambda_A
        self.loss_cycle_B4 = self.criterionCycle(self.rec_B4, self.real_B) * lambda_B
        # self.loss_cycle_C4 = self.criterionCycle(self.rec_C4, self.real_C) * lambda_C
        # self.loss_cycle_D4 = self.criterionCycle(self.rec_D4, self.real_C) * lambda_C
        self.loss_cyc4 = self.loss_cycle_A4 + self.loss_cycle_B4  # + self.loss_cycle_C4 + self.loss_cycle_D4

        self.loss_cycle_temp = self.loss_cyc1 + self.loss_cyc2 + self.loss_cyc3 + self.loss_cyc4

        # Generation loss
        self.loss_PPI_A_temp = self.criterionAE(self.fake_A, self.real_A)
        self.loss_PPI_B_temp = self.criterionAE(self.fake_B, self.real_B)

        # TOTAL loss （这里可以改）
        self.loss_G_temp = self.loss_GABC \
                           + self.loss_AE_temp \
                           + 0.05 * self.loss_cycle_temp

    ########################################################################################################
    def backward_G_with_GCN(self):

        # lambda_A = self.opt.lambda_A
        # lambda_B = self.opt.lambda_B
        # lambda_C = self.opt.lambda_C

        # Autoencoder loss
        # self.fake_A, AErealB1, AErealC1, _ = self.AE.forward_bc2a(self.real_B, self.real_C)
        #
        # AErealA2, self.fake_B, AErealC2, _ = self.AE.forward_ac2b(self.real_A, self.real_C)
        #
        AErealA3, AErealB3, self.fake_C, _ = self.AE.forward_ab2c(self.real_A, self.real_B)

        AErealA4, AErealB4, self.recover_C, _ = self.AE.forward_abc2c(self.real_A, self.real_B, self.real_C)

        # self.loss_AE_A = self.criterionAE(AErealB1, self.real_B) + self.criterionAE(AErealC1, self.real_C)
        #
        # self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + self.criterionAE(AErealC2, self.real_C)
        #
        self.loss_AE_C = self.criterionAE(AErealA3, self.real_A) + self.criterionAE(AErealB3, self.real_B)

        self.loss_AE_D = self.criterionAE(AErealA4, self.real_A) + self.criterionAE(AErealB4, self.real_B)

        # self.loss_AE = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C + self.loss_AE_D
        self.loss_AE = self.loss_AE_C + self.loss_AE_D

        # Generator loss
        # pred_fake = self.netD_A.forward(self.fake_A)
        # self.loss_G_A = self.criterionGAN(pred_fake, True)

        # pred_fake = self.netD_B.forward(self.fake_B)
        # self.loss_G_B = self.criterionGAN(pred_fake, True)

        pred_fake = self.netD_C.forward(self.recover_C)
        self.loss_G_C = self.criterionGAN(pred_fake, True)

        self.loss_GABC = 10 * self.loss_G_C
        # self.loss_GABC = self.loss_G_A + 10 * self.loss_G_C

        # Forward cycle loss
        # self.rec_A1, self.rec_B1, self.rec_C1, _ = self.AE.forward_bc2a(self.real_B, self.recover_C)
        # self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A) * lambda_A
        # self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B) * lambda_B
        # self.loss_cycle_C1 = self.criterionCycle(self.rec_C1, self.real_C) * lambda_C
        # self.loss_cyc1 = self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_cycle_C1
        #
        # self.rec_A2, self.rec_B2, self.rec_C2, _ = self.AE.forward_ac2b(self.fake_A, self.recover_C)
        # # self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B) * lambda_B
        # self.loss_cycle_A2 = self.criterionCycle(self.rec_A2, self.real_A) * lambda_A
        # self.loss_cycle_C2 = self.criterionCycle(self.rec_C2, self.real_C) * lambda_C
        # self.loss_cyc2 = self.loss_cycle_A2 + self.loss_cycle_C2
        #
        # self.rec_A3, self.rec_B3, self.rec_C3, _ = self.AE.forward_ab2c(self.fake_A, self.real_B)
        # self.loss_cycle_C3 = self.criterionCycle(self.rec_C3, self.real_C) * lambda_C
        # self.loss_cycle_A3 = self.criterionCycle(self.rec_A3, self.real_A) * lambda_A
        # self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_B) * lambda_B
        # self.loss_cyc3 = self.loss_cycle_A3 + self.loss_cycle_B3 + self.loss_cycle_C3
        #
        # self.loss_cyc = self.loss_cyc1 + self.loss_cyc2 + self.loss_cyc3

        # PPI loss
        self.loss_PPI_fake = self.criterionAE(self.fake_C, self.real_C)
        self.loss_PPI = self.criterionAE(self.recover_C, self.real_C)
        # self.loss_PPI_A = self.criterionAE(self.fake_A, self.real_A)
        self.loss_G = 20 * self.loss_AE + 2000 * self.loss_PPI + 2000 * self.loss_PPI_fake + 1 * self.loss_GABC  # 这里可以改
        # self.loss_G = 20 * self.loss_AE + 2000 * self.loss_PPI + 2000 * self.loss_PPI_fake + 500 * self.loss_PPI_A + 1 * self.loss_GABC # 这里可以改

        # self.loss_G = 20 * self.loss_AE + 2000 * self.loss_PPI + 2000 * self.loss_PPI_fake + 1 * self.loss_GABC + 10 * self.loss_cyc  # 这里可以改
        self.loss_G_temp = self.loss_G

    def backward_G_with_GCN_unpaired(self):

        # lambda_A = self.opt.lambda_A
        # lambda_B = self.opt.lambda_B
        # lambda_C = self.opt.lambda_C

        # Autoencoder loss
        AErealA3, AErealB3, self.fake_C, _ = self.AE.forward_ab2c(self.real_A, self.real_B)

        AErealA4, AErealB4, self.recover_C, _ = self.AE.forward_abc2c(self.real_A, self.real_B, self.real_C)

        # self.fake_A, AErealB1, AErealC1, _ = self.AE.forward_bc2a(self.real_B, self.recover_C)
        #
        # AErealA2, self.fake_B, AErealC2, _ = self.AE.forward_ac2b(self.real_A, self.recover_C)

        # self.loss_AE_A = self.criterionAE(AErealB1, self.real_B) + self.criterionAE(AErealC1, self.recover_C)
        #
        # self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + self.criterionAE(AErealC2, self.recover_C)

        self.loss_AE_C = self.criterionAE(AErealB3, self.real_B) + self.criterionAE(AErealA3, self.real_A)

        self.loss_AE_D = self.criterionAE(AErealA4, self.real_A) + self.criterionAE(AErealB4, self.real_B)

        self.loss_AE = self.loss_AE_C + self.loss_AE_D
        # self.loss_AE = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C + self.loss_AE_D

        # Generator loss
        # pred_fake = self.netD_A.forward(self.fake_A)
        # self.loss_G_A = self.criterionGAN(pred_fake, True)

        # pred_fake = self.netD_B.forward(self.fake_B)
        # self.loss_G_B = self.criterionGAN(pred_fake, True)

        pred_fake = self.netD_C.forward(self.recover_C)
        self.loss_G_C = self.criterionGAN(pred_fake, True)

        # self.loss_GABC = 0.5 * self.loss_G_A + 0.5 * self.loss_G_B + 2 * self.loss_G_C  # 这里可以改

        self.loss_GABC = 10 * self.loss_G_C  # 这里可以改
        # self.loss_GABC = self.loss_G_A + 10 * self.loss_G_C  # 这里可以改

        # Forward cycle loss
        # self.rec_A1, self.rec_B1, self.rec_C1, _ = self.AE.forward_bc2a(self.real_B, self.recover_C)
        # self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A) * lambda_A
        # self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B) * lambda_B
        # # self.loss_cycle_C1 = self.criterionCycle(self.rec_C1, self.fake_C) * lambda_C
        # self.loss_cyc1 = self.loss_cycle_A1 + self.loss_cycle_B1  # + self.loss_cycle_B1
        #
        # self.rec_A2, self.rec_B2, self.rec_C2, _ = self.AE.forward_ac2b(self.fake_A, self.recover_C)
        # # self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B) * lambda_B
        # self.loss_cycle_A2 = self.criterionCycle(self.rec_A2, self.real_A) * lambda_A
        # # self.loss_cycle_C2 = self.criterionCycle(self.rec_C2, self.fake_C) * lambda_C
        # self.loss_cyc2 = self.loss_cycle_A2 #+ self.loss_cycle_B2  # + self.loss_cycle_C2
        #
        # self.rec_A3, self.rec_B3, self.rec_C3, _ = self.AE.forward_ab2c(self.fake_A, self.real_B)
        # self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_B) * lambda_B
        # self.loss_cycle_A3 = self.criterionCycle(self.rec_A3, self.real_A) * lambda_A
        # # self.loss_cycle_C3 = self.criterionCycle(self.rec_C3, self.fake_C) * lambda_C
        # self.loss_cyc3 = self.loss_cycle_A3 + self.loss_cycle_B3  # + self.loss_cycle_C3
        #
        # self.loss_cyc = self.loss_cyc1 + self.loss_cyc2 + self.loss_cyc3

        # self.loss_PPI_A = self.criterionAE(self.fake_A, self.real_A)

        self.loss_G = 20 * self.loss_AE + 1 * self.loss_GABC  # 这里可以改
        # self.loss_G = 20 * self.loss_AE + 1 * self.loss_GABC + 500 * self.loss_PPI_A # 这里可以改
        # self.loss_G = 20 * self.loss_AE + 1 * self.loss_GABC + 500 * loss_PPI_A + 10 * self.loss_cyc  # 这里可以改

        self.loss_G_temp = self.loss_G

        self.PPI_temp = self.recover_C
        self.PPI_temp_fake = self.fake_C

        # self.cycle_C1 = self.rec_C1
        # self.cycle_C2 = self.rec_C2
        # self.cycle_C3 = self.rec_C3

    def backward_GCN_with_G(self):

        self.GCN_out_A = self.C1.forward(self.E1.forward(self.A_matrix, self.A_adj))
        self.GCN_out_B = self.C2.forward(self.E2.forward(self.B_matrix, self.B_adj))
        self.GCN_out_C = self.C3.forward(self.E3.forward(self.C_matrix, self.C_adj))

        gene_pred_A = []
        gene_pred_B = []
        gene_pred_C = []

        for gene_idx in range(self.iso_gene_unpaired[0, 1], self.iso_gene_unpaired[-1, 1] + 1):
            iso_idx = self.iso_gene_unpaired[torch.where(self.iso_gene_unpaired[:, 1] == gene_idx)[0], 0]
            gene_pred_A.append(torch.max(self.GCN_out_A[iso_idx, :], dim=0)[0].view(1, self.GCN_out_A.shape[-1]))
            gene_pred_B.append(torch.max(self.GCN_out_B[iso_idx, :], dim=0)[0].view(1, self.GCN_out_B.shape[-1]))
            gene_pred_C.append(torch.max(self.GCN_out_C[iso_idx, :], dim=0)[0].view(1, self.GCN_out_C.shape[-1]))

        self.fpred_A = torch.cat([self.GCN_out_A[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_A, dim=0)],
                                 dim=0)  # 这
        self.fpred_B = torch.cat([self.GCN_out_B[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_B, dim=0)],
                                 dim=0)
        self.fpred_C = torch.cat([self.GCN_out_C[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_C, dim=0)],
                                 dim=0)

        self.loss_GCN_A = torch.mean(torch.mul(self.criterionGCN(self.fpred_A, self.labels), self.labels_weight))
        self.loss_GCN_B = torch.mean(torch.mul(self.criterionGCN(self.fpred_B, self.labels), self.labels_weight))
        self.loss_GCN_C = torch.mean(torch.mul(self.criterionGCN(self.fpred_C, self.labels), self.labels_weight))

        self.loss_GCN = self.loss_GCN_A + self.loss_GCN_B + self.loss_GCN_C


    def backward_GCN_with_VCDN(self):

        self.GCN_out_A = self.C1.forward(self.E1.forward(self.A_matrix, self.A_adj))
        self.GCN_out_B = self.C2.forward(self.E2.forward(self.B_matrix, self.B_adj))
        self.GCN_out_C = self.C3.forward(self.E3.forward(self.C_matrix, self.C_adj))

        gene_pred_A = []
        gene_pred_B = []
        gene_pred_C = []

        for gene_idx in range(self.iso_gene_unpaired[0, 1], self.iso_gene_unpaired[-1, 1] + 1):
            iso_idx = self.iso_gene_unpaired[torch.where(self.iso_gene_unpaired[:, 1] == gene_idx)[0], 0]
            gene_pred_A.append(torch.max(self.GCN_out_A[iso_idx, :], dim=0)[0].view(1, self.GCN_out_A.shape[-1]))
            gene_pred_B.append(torch.max(self.GCN_out_B[iso_idx, :], dim=0)[0].view(1, self.GCN_out_B.shape[-1]))
            gene_pred_C.append(torch.max(self.GCN_out_C[iso_idx, :], dim=0)[0].view(1, self.GCN_out_C.shape[-1]))

        self.fpred_A = torch.cat([self.GCN_out_A[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_A, dim=0)],
                                 dim=0)
        self.fpred_B = torch.cat([self.GCN_out_B[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_B, dim=0)],
                                 dim=0)
        self.fpred_C = torch.cat([self.GCN_out_C[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_C, dim=0)],
                                 dim=0)

        self.loss_GCN_A = torch.mean(torch.mul(self.criterionGCN(self.fpred_A, self.labels), self.labels_weight))
        self.loss_GCN_B = torch.mean(torch.mul(self.criterionGCN(self.fpred_B, self.labels), self.labels_weight))
        self.loss_GCN_C = torch.mean(torch.mul(self.criterionGCN(self.fpred_C, self.labels), self.labels_weight))

        self.loss_GCN_A.backward()
        self.loss_GCN_B.backward()
        self.loss_GCN_C.backward()

    ####################################################################################################################

    def backward_GCN_VCDN(self):

        self.GCN_out_A = self.C1.forward(self.E1.forward(self.A_matrix, self.A_adj))
        self.GCN_out_B = self.C2.forward(self.E2.forward(self.B_matrix, self.B_adj))
        self.GCN_out_C = self.C3.forward(self.E3.forward(self.C_matrix, self.C_adj))

        gene_pred_A = []
        gene_pred_B = []
        gene_pred_C = []

        for gene_idx in range(self.iso_gene_unpaired[0, 1], self.iso_gene_unpaired[-1, 1] + 1):
            iso_idx = self.iso_gene_unpaired[torch.where(self.iso_gene_unpaired[:, 1] == gene_idx)[0], 0]
            gene_pred_A.append(torch.max(self.GCN_out_A[iso_idx, :], dim=0)[0].view(1, self.GCN_out_A.shape[-1]))
            gene_pred_B.append(torch.max(self.GCN_out_B[iso_idx, :], dim=0)[0].view(1, self.GCN_out_B.shape[-1]))
            gene_pred_C.append(torch.max(self.GCN_out_C[iso_idx, :], dim=0)[0].view(1, self.GCN_out_C.shape[-1]))

        self.fpred_A = torch.cat([self.GCN_out_A[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_A, dim=0)],
                                 dim=0)
        self.fpred_B = torch.cat([self.GCN_out_B[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_B, dim=0)],
                                 dim=0)
        self.fpred_C = torch.cat([self.GCN_out_C[0:self.iso_gene_paired.shape[0], :], torch.cat(gene_pred_C, dim=0)],
                                 dim=0)

        self.loss_GCN_A = torch.mean(torch.mul(self.criterionGCN(self.fpred_A, self.labels), self.labels_weight))
        self.loss_GCN_B = torch.mean(torch.mul(self.criterionGCN(self.fpred_B, self.labels), self.labels_weight))
        self.loss_GCN_C = torch.mean(torch.mul(self.criterionGCN(self.fpred_C, self.labels), self.labels_weight))

        self.VCDN_input = [self.GCN_out_A, self.GCN_out_B, self.GCN_out_C]
        # self.VCDN_input = [self.GCN_out_A, self.GCN_out_B]
        # self.VCDN_out,_ = self.VCDN.forward(self.VCDN_input)
        _, self.VCDN_out = self.VCDN.forward(self.VCDN_input)

        gene_pred_VCDN = []

        for gene_idx in range(self.iso_gene_unpaired[0, 1], self.iso_gene_unpaired[-1, 1] + 1):
            iso_idx = self.iso_gene_unpaired[torch.where(self.iso_gene_unpaired[:, 1] == gene_idx)[0], 0]
            gene_pred_VCDN.append(torch.max(self.VCDN_out[iso_idx, :], dim=0)[0].view(1, self.VCDN_out.shape[-1]))

        self.fpred_VCDN = torch.cat([self.VCDN_out[0:len(self.iso_gene_paired), :], torch.cat(gene_pred_VCDN, dim=0)],
                                    dim=0)
        self.loss_VCDN = torch.mean(torch.mul(self.criterionVCDN(self.fpred_VCDN, self.labels), self.labels_weight))

        self.loss_VCDN.backward()

    ####################################################################################################################

    ############################################################################
    # Define optimize function for VIGAN
    ############################################################################

    def optimize_pretrain_AE(self):
        # forward
        self.forward()
        # AE
        self.optimizer_AE.zero_grad()
        self.backward_AE_pretrain()
        self.optimizer_AE.step()

    def test_pretrain_AE(self):
        # forward
        self.forward()
        # AE
        with torch.no_grad():
            self.test_AE_pretrain()

    #############################################################################

    def optimize_pretrain_cycleGAN(self):
        self.forward()

        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_C
        self.optimizer_D_C.zero_grad()
        self.backward_D_C()
        self.optimizer_D_C.step()

    def optimize_pretrain_cycleGAN_unpaired(self, init_iso_of_gene):
        # forward
        self.forward()

        # G
        self.backward_G_unpaired()

        # D
        # self.backward_D_A_unpaired()
        # self.backward_D_B_unpaired()
        self.backward_D_C_unpaired()

        if init_iso_of_gene:
            # TOTAL
            self.loss_G_temp_batch = [self.loss_G_temp]

            # AE
            self.loss_AE_temp_batch = [self.loss_AE_temp]
            self.AErealC1_batch = [self.AErealC1]
            self.AErealC2_batch = [self.AErealC2]
            self.AErealC4_batch = [self.AErealC4]

            # cycle
            self.loss_cycle_temp_batch = [self.loss_cycle_temp]
            self.cycle_C1_batch = [self.rec_C1]
            self.cycle_C2_batch = [self.rec_C2]
            self.cycle_C3_batch = [self.rec_C3]
            self.cycle_C4_batch = [self.rec_C4]
            self.cycle_D4_batch = [self.rec_D4]

            # Generation
            self.loss_PPI_A_temp_batch = [self.loss_PPI_A_temp]
            self.loss_PPI_B_temp_batch = [self.loss_PPI_B_temp]
            self.PPI_fake_batch = [self.fake_C]
            self.PPI_batch = [self.fake_D]

            # GAN
            # self.loss_G_A_batch = [self.loss_G_A]
            # self.loss_G_B_batch = [self.loss_G_B]
            self.loss_G_C_batch = [self.loss_G_C]
            # self.loss_D_A_batch = [self.loss_D_A]
            # self.loss_D_B_batch = [self.loss_D_B]
            self.loss_D_C_batch = [self.loss_D_C]


        else:
            # TOTAL
            self.loss_G_temp_batch += [self.loss_G_temp]

            # AE
            self.loss_AE_temp_batch += [self.loss_AE_temp]
            self.AErealC1_batch += [self.AErealC1]
            self.AErealC2_batch += [self.AErealC2]
            self.AErealC4_batch += [self.AErealC4]

            # cycle
            self.loss_cycle_temp_batch += [self.loss_cycle_temp]
            self.cycle_C1_batch += [self.rec_C1]
            self.cycle_C2_batch += [self.rec_C2]
            self.cycle_C3_batch += [self.rec_C3]
            self.cycle_C4_batch += [self.rec_C4]
            self.cycle_D4_batch += [self.rec_D4]

            # Generation
            self.loss_PPI_A_temp_batch += [self.loss_PPI_A_temp]
            self.loss_PPI_B_temp_batch += [self.loss_PPI_B_temp]
            self.PPI_fake_batch += [self.fake_C]
            self.PPI_batch += [self.fake_D]

            # GAN
            # self.loss_G_A_batch = [self.loss_G_A]
            # self.loss_G_B_batch = [self.loss_G_B]
            self.loss_G_C_batch += [self.loss_G_C]
            # self.loss_D_A_batch = [self.loss_D_A]
            # self.loss_D_B_batch = [self.loss_D_B]
            self.loss_D_C_batch += [self.loss_D_C]

    def optimize_pretrain_cycleGAN_unpaired_batch(self, gene_PPI):

        gene_PPI = gene_PPI.to(self.device)

        self.iso_count = len(self.PPI_batch)

        # TOTAL
        self.loss_G_temp_batch = sum(self.loss_G_temp_batch) / self.iso_count

        # AE
        self.loss_AE_temp_batch = sum(self.loss_AE_temp_batch) / self.iso_count

        self.AErealC1_batch = torch.max(torch.cat(self.AErealC1_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)
        self.AErealC2_batch = torch.max(torch.cat(self.AErealC2_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)
        self.AErealC4_batch = torch.max(torch.cat(self.AErealC4_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)

        class_weights = self.class_weights_for_PPI_input(gene_PPI.flatten())
        self.criterionAE_C = torch.nn.BCELoss(weight=class_weights)
        # self.criterionCycle_C = torch.nn.BCELoss(weight=self.class_weights)

        self.loss_AErealC = self.criterionAE_C(self.AErealC1_batch, gene_PPI) \
                            + self.criterionAE_C(self.AErealC2_batch, gene_PPI) \
                            + self.criterionAE_C(self.AErealC4_batch, gene_PPI)

        self.loss_AE_batch = self.loss_AE_temp_batch + self.loss_AErealC

        # cycle
        self.loss_cycle_temp_batch = sum(self.loss_cycle_temp_batch) / self.iso_count

        self.cycle_C1_batch = torch.max(torch.cat(self.cycle_C1_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)
        self.cycle_C2_batch = torch.max(torch.cat(self.cycle_C2_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)
        self.cycle_C3_batch = torch.max(torch.cat(self.cycle_C3_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)
        self.cycle_C4_batch = torch.max(torch.cat(self.cycle_C4_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)
        self.cycle_D4_batch = torch.max(torch.cat(self.cycle_D4_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)

        lambda_C = self.opt.lambda_C
        self.cycle_loss_C = lambda_C * (self.criterionCycle_C(self.cycle_C1_batch, gene_PPI) +
                                        self.criterionCycle_C(self.cycle_C2_batch, gene_PPI) +
                                        self.criterionCycle_C(self.cycle_C3_batch, gene_PPI) +
                                        self.criterionCycle_C(self.cycle_C4_batch, gene_PPI) +
                                        self.criterionCycle_C(self.cycle_D4_batch, gene_PPI))

        self.loss_cycle_batch = self.loss_cycle_temp_batch + self.cycle_loss_C

        # Generation
        self.loss_PPI_A_batch = sum(self.loss_PPI_A_temp_batch) / self.iso_count
        self.loss_PPI_B_batch = sum(self.loss_PPI_B_temp_batch) / self.iso_count

        self.PPI_fake_batch = torch.max(torch.cat(self.PPI_fake_batch, dim=0), dim=0)[0].view(1, 1,
                                                                                              self.gene_PPI_column)
        self.PPI_batch = torch.max(torch.cat(self.PPI_batch, dim=0), dim=0)[0].view(1, 1, self.gene_PPI_column)

        self.loss_PPI_fake_batch = self.criterionAE_C(self.PPI_fake_batch, gene_PPI)
        self.loss_PPI_batch = self.criterionAE_C(self.PPI_batch, gene_PPI)

        self.loss_AE_batch = self.loss_AE_batch + self.loss_PPI_fake_batch + self.loss_PPI_batch

        # GAN
        # self.loss_G_A_batch = sum(self.loss_G_A_batch) / self.iso_count
        # self.loss_G_B_batch = sum(self.loss_G_B_batch) / self.iso_count
        self.loss_G_C_batch = sum(self.loss_G_C_batch) / self.iso_count
        # self.loss_D_A_batch = sum(self.loss_D_A_batch) / self.iso_count
        # self.loss_D_B_batch = sum(self.loss_D_B_batch) / self.iso_count
        self.loss_D_C_batch = sum(self.loss_D_C_batch) / self.iso_count

        self.loss_G_batch = self.loss_G_temp_batch \
                            + self.loss_PPI_batch \
                            + self.loss_PPI_fake_batch \
                            + self.loss_AErealC \
                            + 0.05 * self.cycle_loss_C  # 可以修改！

        self.optimizer_G.zero_grad()
        self.loss_G_batch.backward()
        self.optimizer_G.step()

        # self.optimizer_D_A.zero_grad()
        # self.loss_D_A_batch.backward()
        # self.optimizer_D_A.step()
        # D_B
        # self.optimizer_D_B.zero_grad()
        # self.loss_D_B_batch.backward()
        # self.optimizer_D_B.step()
        # D_C
        self.optimizer_D_C.zero_grad()
        self.loss_D_C_batch.backward()
        self.optimizer_D_C.step()

    def test_pretrain_cycleGAN_unpaired_batch(self, gene_PPI):

        self.iso_count = len(self.PPI_batch)

        self.loss_PPI_A_batch = sum(self.loss_PPI_A_batch) / self.iso_count

        self.loss_G_temp_batch = sum(self.loss_G_temp_batch) / self.iso_count

        self.loss_cycle_temp_batch = sum(self.loss_cycle_temp_batch) / self.iso_count

        self.loss_D_A_batch = sum(self.loss_D_A_batch) / self.iso_count
        # self.loss_D_B_batch = sum(self.loss_D_B_batch) / self.iso_count
        self.loss_D_C_batch = sum(self.loss_D_C_batch) / self.iso_count

        self.loss_D_A_f_batch = sum(self.loss_D_A_f_batch) / self.iso_count
        # self.loss_D_B_f_batch = sum(self.loss_D_B_f_batch) / self.iso_count
        self.loss_D_C_f_batch = sum(self.loss_D_C_f_batch) / self.iso_count

        self.loss_G_A_batch = sum(self.loss_G_A_batch) / self.iso_count
        # self.loss_G_B_batch = sum(self.loss_G_B_batch) / self.iso_count
        self.loss_G_C_batch = sum(self.loss_G_C_batch) / self.iso_count

        self.PPI_batch = torch.max(torch.cat(self.PPI_batch, dim=0), dim=0)[0]
        num_column = self.PPI_batch.shape[-1]
        self.PPI_batch = self.PPI_batch.view(1, 1, num_column)

        self.PPI_fake_batch = torch.max(torch.cat(self.PPI_fake_batch, dim=0), dim=0)[0].view(1, 1, num_column)
        self.cycle_C1_batch = torch.max(torch.cat(self.cycle_C1_batch, dim=0), dim=0)[0].view(1, 1, num_column)
        self.cycle_C2_batch = torch.max(torch.cat(self.cycle_C2_batch, dim=0), dim=0)[0].view(1, 1, num_column)
        self.cycle_C3_batch = torch.max(torch.cat(self.cycle_C3_batch, dim=0), dim=0)[0].view(1, 1, num_column)

        gene_PPI = gene_PPI  # .cuda(self.gpu_ids[0])
        self.loss_PPI = self.criterionAE(self.PPI_batch, gene_PPI)
        self.loss_PPI_fake = self.criterionAE(self.PPI_fake_batch, gene_PPI)

        lambda_C = self.opt.lambda_C
        self.cycle_loss_C = lambda_C * (self.criterionCycle(self.cycle_C1_batch, gene_PPI) +
                                        self.criterionCycle(self.cycle_C2_batch, gene_PPI) +
                                        self.criterionCycle(self.cycle_C3_batch, gene_PPI))

        self.loss_G_batch = self.loss_G_temp_batch + 2000 * self.loss_PPI + 2000 * self.loss_PPI_fake + 10 * self.cycle_loss_C  # 可以修改！

        # self.loss_G = self.loss_GABC + 20 * self.loss_AE + 10 * self.loss_cyc1 + 10 * self.loss_cyc2 + 10 * self.loss_cyc3  # 这里可以改
        self.loss_cycle = self.loss_cycle_temp_batch + self.cycle_loss_C

    #############################################################################

    def optimize_cycleGAN_GCN(self, init):
        # forward
        self.forward()

        self.backward_G_with_GCN()

        # self.backward_D_A_unpaired()
        # self.backward_D_B_unpaired()
        self.backward_D_C_unpaired()

        if init:
            # G
            self.loss_G_all = [self.loss_G_temp]

            # PPI
            self.loss_PPI_all = [self.loss_PPI]
            self.loss_PPI_fake_all = [self.loss_PPI_fake]

            # D_A
            # self.loss_D_A_all = [self.loss_D_A]
            # D_B
            # self.loss_D_B_all = [self.backward_D_B_unpaired()]
            # D_C
            self.loss_D_C_all = [self.loss_D_C]
        else:
            # G
            self.loss_G_all += [self.loss_G_temp]

            # PPI
            self.loss_PPI_all += [self.loss_PPI]
            self.loss_PPI_fake_all += [self.loss_PPI_fake]

            # D_A
            # self.loss_D_A_all += [self.loss_D_A]
            # D_B
            # self.loss_D_B_all += [self.backward_D_B_unpaired()]
            # D_C
            self.loss_D_C_all += [self.loss_D_C]

    def optimize_cycleGAN_GCN_unpaired(self, init_iso_of_the_gene, init_iso_of_unpaired_gene):

        # forward
        self.forward()

        self.backward_G_with_GCN_unpaired()

        # self.backward_D_A_unpaired()
        self.backward_D_C_unpaired()

        if init_iso_of_the_gene:
            # G
            self.loss_G_batch = [self.loss_G_temp]

            # D_A
            # self.loss_D_A_batch = [self.loss_D_A]
            # D_B
            # self.loss_D_B_batch = [self.backward_D_B_unpaired()]
            # D_C
            self.loss_D_C_batch = [self.loss_D_C]

            self.PPI_batch = [self.PPI_temp]
            self.PPI_fake_batch = [self.PPI_temp_fake]
            # self.cycle_C1_batch = [self.cycle_C1]
            # self.cycle_C2_batch = [self.cycle_C2]
            # self.cycle_C3_batch = [self.cycle_C3]

        else:
            # G
            self.loss_G_batch += [self.loss_G_temp]

            # D_A
            # self.loss_D_A_batch += [self.loss_D_A]
            # D_B
            # self.loss_D_B_batch += [self.backward_D_B_unpaired()]
            # D_C
            self.loss_D_C_batch += [self.loss_D_C]

            self.PPI_batch += [self.PPI_temp]
            self.PPI_fake_batch += [self.PPI_temp_fake]
            # self.cycle_C1_batch += [self.cycle_C1]
            # self.cycle_C2_batch += [self.cycle_C2]
            # self.cycle_C3_batch += [self.cycle_C3]

        if init_iso_of_unpaired_gene:
            self.PPI_unpaired = [self.PPI_temp]
        else:
            self.PPI_unpaired += [self.PPI_temp]

    def optimize_cycleGAN_GCN_batch(self, gene_PPI):

        self.iso_count = len(self.PPI_batch)

        # Loss1
        self.loss_G_batch = sum(self.loss_G_batch) / self.iso_count

        # self.loss_D_A_batch = sum(self.loss_D_A_batch) / self.iso_count
        # self.loss_D_B_batch = sum(self.loss_D_B_batch) / self.iso_count
        self.loss_D_C_batch = sum(self.loss_D_C_batch) / self.iso_count

        self.PPI_batch = torch.max(torch.cat(self.PPI_batch, dim=0), dim=0)[0]
        num_column = self.PPI_batch.shape[-1]
        self.PPI_batch = self.PPI_batch.view(1, 1, num_column)

        self.PPI_fake_batch = torch.max(torch.cat(self.PPI_fake_batch, dim=0), dim=0)[0].view(1, 1, num_column)
        # self.cycle_C1_batch = torch.max(torch.cat(self.cycle_C1_batch, dim=0), dim=0)[0].view(1, 1, num_column)
        # self.cycle_C2_batch = torch.max(torch.cat(self.cycle_C2_batch, dim=0), dim=0)[0].view(1, 1, num_column)
        # self.cycle_C3_batch = torch.max(torch.cat(self.cycle_C3_batch, dim=0), dim=0)[0].view(1, 1, num_column)

        gene_PPI = gene_PPI  # .cuda(self.gpu_ids[0])

        self.loss_PPI = self.criterionAE(self.PPI_batch, gene_PPI)
        self.loss_PPI_fake = self.criterionAE(self.PPI_fake_batch, gene_PPI)

        # lambda_C = self.opt.lambda_C
        # self.cycle_loss = lambda_C * (self.criterionCycle(self.cycle_C1_batch, gene_PPI) +
        #                               self.criterionCycle(self.cycle_C2_batch, gene_PPI) +
        #                               self.criterionCycle(self.cycle_C3_batch, gene_PPI))

        self.loss_G_batch = self.loss_G_batch + 2000 * self.loss_PPI + 2000 * self.loss_PPI_fake

        # self.loss_G_batch = self.loss_G_batch + 2000 * self.loss_PPI + 2000 * self.loss_PPI_fake + 10 * self.cycle_loss

        # self.loss_G = 10 * self.loss_AE + 1 * self.loss_GABC + 1 * self.loss_cyc #这里可以改

        # combine loss
        # G
        self.loss_G_all += [self.loss_G_batch]
        # PPI
        self.loss_PPI_all += [self.loss_PPI]
        self.loss_PPI_fake_all += [self.loss_PPI_fake]
        # D_A
        # self.loss_D_A_all += [self.loss_D_A_batch]
        # D_B
        # self.loss_D_B_all += [self.loss_D_B_batch]
        # D_C
        self.loss_D_C_all += [self.loss_D_C_batch]

    def optimize_cycleGAN_GCN_all(self):


        self.forward_AE_GCN()

        self.E1.train()
        self.E2.train()
        self.E3.train()
        self.C1.train()
        self.C2.train()
        self.C3.train()

        self.backward_GCN_with_G()

        print('start cycleGAN GCN backward')
        self.optimizer_E1_C1_pretrain.zero_grad()
        self.optimizer_E2_C2_pretrain.zero_grad()
        self.optimizer_AE_GCN_C.zero_grad()
        self.loss_GCN_A.backward()
        self.loss_GCN_B.backward()
        self.loss_GCN_C.backward()
        self.optimizer_E1_C1_pretrain.step()
        self.optimizer_E2_C2_pretrain.step()
        self.optimizer_AE_GCN_C.step()

        # self.optimizer_D_A.zero_grad()
        # self.loss_D_A_all.backward()
        # self.optimizer_D_A.step()
        # D_B
        # self.optimizer_D_B.zero_grad()
        # self.loss_D_B_all.backward()
        # self.optimizer_D_B.step()
        # D_C

        print('finish cycleGAN GCN backward')

    def test_cycleGAN_GCN(self):

        self.E1.eval()
        self.E2.eval()
        self.E3.eval()
        self.C1.eval()
        self.C2.eval()
        self.C3.eval()

        # forward
        self.forward_AE_GCN_test()
        self.GCN_out_A = self.C1.forward(self.E1.forward(self.A_matrix, self.A_adj))
        self.GCN_out_B = self.C2.forward(self.E2.forward(self.B_matrix, self.B_adj))
        self.GCN_out_C = self.C3.forward(self.E3.forward(self.C_matrix, self.C_adj))

        return torch.sigmoid(self.GCN_out_A), torch.sigmoid(self.GCN_out_B), torch.sigmoid(self.GCN_out_C)

    #############################################################################

    def optimize_GCN_VCDN(self):

        self.E1.train()
        self.E2.train()
        self.E3.train()
        self.C1.train()
        self.C2.train()
        self.C3.train()
        self.VCDN.train()

        # forward
        self.forward_GCN_VCDN()

        self.optimizer_E1_C1.zero_grad()
        self.optimizer_E2_C2.zero_grad()
        self.optimizer_E3_C3.zero_grad()

        self.backward_GCN_with_VCDN()

        self.optimizer_E1_C1.step()
        self.optimizer_E2_C2.step()
        self.optimizer_E3_C3.step()

        self.optimizer_GCN_VCDN.zero_grad()
        self.backward_GCN_VCDN()
        self.optimizer_GCN_VCDN.step()

    #############################################################################

    def optimize_VCDN(self):

        self.VCDN.train()

        # forward
        self.forward_GCN_VCDN()

        self.optimizer_GCN_VCDN.zero_grad()
        self.backward_GCN_VCDN()
        self.optimizer_GCN_VCDN.step()

    #############################################################################

    def test_GCN_VCDN(self):

        self.E1.eval()
        self.E2.eval()
        self.E3.eval()
        self.C1.eval()
        self.C2.eval()
        self.C3.eval()

        self.VCDN.eval()

        # forward
        self.forward_GCN_VCDN()

        self.GCN_out_A = self.C1.forward(self.E1.forward(self.A_matrix, self.A_adj))
        self.GCN_out_B = self.C2.forward(self.E2.forward(self.B_matrix, self.B_adj))
        self.GCN_out_C = self.C3.forward(self.E3.forward(self.C_matrix, self.C_adj))

        self.VCDN_input = [self.GCN_out_A, self.GCN_out_B, self.GCN_out_C]
        self.VCDN_out, self.VCDN_prob = self.VCDN.forward(self.VCDN_input)

        return self.VCDN_out, self.VCDN_prob

    # def optimize_AECL(self):
    #     # forward
    #     self.forward()
    #
    #     self.optimizer_AE_CL.zero_grad()
    #     self.backward_AE_CL()
    #     self.optimizer_AE_CL.step()

    #        for i in range(1):
    #            # D_A
    #            self.optimizer_D_A.zero_grad()
    #            self.backward_D_A()
    #            self.optimizer_D_A.step()
    #            # D_B
    #            self.optimizer_D_B.zero_grad()
    #            self.backward_D_B()
    #            self.optimizer_D_B.step()
    #     def optimize_parameters(self):
    #         # forward
    #
    #         self.forward()
    # #        self.optimizer_AE_GA_GB.zero_grad()
    # #        self.backward_AE_GA_GB()
    # #        self.optimizer_AE_GA_GB.step()
    #         # AE+G_A+G_B
    #
    #         for i in range(2):
    #             self.optimizer_AE_GA_GB.zero_grad()
    #             self.backward_AE_GA_GB()
    #             self.optimizer_AE_GA_GB.step()
    #
    #         for i in range(1):
    #             # D_A
    #             self.optimizer_D_A_AE.zero_grad()
    #             self.backward_D_A_AE()
    #             self.optimizer_D_A_AE.step()
    #             # D_B
    #             self.optimizer_D_B_AE.zero_grad()
    #             self.backward_D_B_AE()
    #             self.optimizer_D_B_AE.step()

    ############################################################################################
    # Get errors for visualization
    ############################################################################################

    def get_current_errors_cycle_pre(self):

        # Discriminator judgement loss
        AE_D_C = self.loss_D_C.item()

        # Generator prediction loss
        AE_G_C = self.loss_G_C.item()

        # AutoEncoder recovery loss
        AE = self.loss_AE.item()

        # PPI generation loss
        PPI = self.loss_PPI.item()
        PPI_fake = self.loss_PPI_fake.item()
        PPI_A = self.loss_PPI_A.item()
        PPI_B = self.loss_PPI_B.item()

        # Generator loss + AutoEncoder recovery loss + PPI generation loss + cycle GAN (recovery + generation) loss
        ALL_loss = self.loss_G.item()

        cycle_loss = self.loss_cycle.item()


        return OrderedDict([  # ('G_A', AE_G_A),
            # ('G_B', AE_G_B),
            ('G_C', AE_G_C),
            # ('D_A', AE_D_A),
            # ('D_B', AE_D_B),
            ('D_C', AE_D_C),
            # ('AE_A', AE_A),
            # ('AE_B', AE_B),
            # ('AE_C', AE_C),
            ('AE', AE),
            ('cycle', cycle_loss),
            ('PPI', PPI), ('PPI_fake', PPI_fake), ('expr', PPI_A), ('seqdm', PPI_B), ('ALL', ALL_loss)])

    def get_current_errors_cycle_pre_unpaired(self):

        # Discriminator judgement loss
        # AE_D_A = self.loss_D_A_batch.item()
        # AE_D_B = self.loss_D_B_f_batch.item()
        AE_D_C = self.loss_D_C_batch.item()

        # AE_G_A = self.loss_G_A_batch.item()
        # AE_G_B = self.loss_G_B_batch.item()
        AE_G_C = self.loss_G_C_batch.item()

        AE = self.loss_AE_batch.item()

        # PPI generation loss
        PPI = self.loss_PPI_batch.item()
        PPI_fake = self.loss_PPI_fake_batch.item()
        PPI_A = self.loss_PPI_A_batch.item()
        PPI_B = self.loss_PPI_B_batch.item()
        # Generator prediction loss + AutoEncoder recovery loss + PPI generation loss + cycle GAN (recovery + generate (except PPI)) loss + cycle GAN (recovery + generate（PPI)) loss
        ALL_loss = self.loss_G_batch.item()

        cycle_loss = self.loss_cycle_batch.item()


        return OrderedDict([
            ('G_C', AE_G_C),
            ('D_C', AE_D_C),
            ('AE', AE),
            ('PPI', PPI), ('PPI_fake', PPI_fake), ('cycle', cycle_loss), ('expr', PPI_A), ('seqdm', PPI_B),
            ('ALL', ALL_loss)])

    def get_current_errors_cycleGAN_GCN(self):

        # GCN_loss
        GCN_A = self.loss_GCN_A.item()
        GCN_B = self.loss_GCN_B.item()
        GCN_C = self.loss_GCN_C.item()
        GCN_all = self.loss_GCN.item()

        return OrderedDict([
                ('GCN_A', GCN_A), ('GCN_B', GCN_B), ('GCN_C', GCN_C),
                ('GCN_all', GCN_all)])


    def get_current_errors_GCN_VCDN(self):

        GCN_A = self.loss_GCN_A.item()
        GCN_B = self.loss_GCN_B.item()
        GCN_C = self.loss_GCN_C.item()
        VCDN = self.loss_VCDN.item()

        return OrderedDict([('GCN_A', GCN_A), ('GCN_B', GCN_B), ('GCN_C', GCN_C),
                                ('VCDN', VCDN)])

    #     def get_current_errors_AE_CL(self):
    # #        AE = self.loss_AE.item()
    #         CLU_loss = self.loss_clustering.item()
    #         AE = self.loss_AE.item()
    #         if self.opt.identity > 0.0:
    #             return OrderedDict([('CLU', CLU_loss), ('AE', AE)])
    #         else:
    #             return OrderedDict([ ('CLU', CLU_loss), ('AE', AE)])
    #
    #
    #
    #
    #
    #
    #     def get_current_errors(self):
    #         D_A = self.loss_D_A_AE.item()
    #         G_A = self.loss_AE_GA.item()
    #         Cyc_A = self.loss_cycle_A_AE.item()
    #         D_B = self.loss_D_B_AE.item()
    #         G_B = self.loss_AE_GB.item()
    #         Cyc_B = self.loss_cycle_B_AE.item()
    #         clu_loss = self.clustering_loss.item()
    #         loss_all = self.loss_AE_GA_GB.item()
    #         if self.opt.identity > 0.0:
    #             idt_A = self.loss_idt_A.item()
    #             idt_B = self.loss_idt_B.item()
    #             return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
    #                                 ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
    #         else:
    #             return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),('loss_all',loss_all),
    #                                 ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),('clu_loss',clu_loss)])

    # def get_current_visuals(self):
    #     real_A = util.tensor2im(self.real_A.data)
    #     fake_B = util.tensor2im(self.fake_B.data)
    #     rec_A  = util.tensor2im(self.rec_A.data)
    #     real_B = util.tensor2im(self.real_B.data)
    #     fake_A = util.tensor2im(self.fake_A.data)
    #     rec_B  = util.tensor2im(self.rec_B.data)
    #
    #     AE_fake_A = util.tensor2im(self.AEfakeA.view(1,1,28,28).data)
    #     AE_fake_B = util.tensor2im(self.AEfakeB.view(1,1,28,28).data)
    #
    #
    #     if self.opt.identity > 0.0:
    #         idt_A = util.tensor2im(self.idt_A.data)
    #         idt_B = util.tensor2im(self.idt_B.data)
    #         return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
    #                             ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A),
    #                             ('AE_fake_A', AE_fake_A), ('AE_fake_B', AE_fake_B)])
    #     else:
    #         return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
    #                             ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
    #                             ('AE_fake_A', AE_fake_A), ('AE_fake_B', AE_fake_B)])
    #
    def save(self, i):
        self.save_network(self.netD_C, 'D_C', i, self.gpu_ids)
        self.save_network(self.AE, 'AE', i, self.gpu_ids)

    def save_GCN(self, i='after_GCN_pretrain'):
        self.save_network(self.C1, 'C1', i, self.gpu_ids,self.tissue)
        self.save_network(self.E1, 'E1', i, self.gpu_ids,self.tissue)
        self.save_network(self.C2, 'C2', i, self.gpu_ids,self.tissue)
        self.save_network(self.E2, 'E2', i, self.gpu_ids,self.tissue)
        self.save_network(self.C3, 'C3', i, self.gpu_ids,self.tissue)
        self.save_network(self.E3, 'E3', i, self.gpu_ids,self.tissue)

        # self.save_network(self.netD_A, 'D_A', i, self.gpu_ids)
        # self.save_network(self.netD_B, 'D_B', i, self.gpu_ids)
        # self.save_network(self.netD_C, 'D_C', i, self.gpu_ids)
        # self.save_network(self.AE, 'AE', i, self.gpu_ids)

#
# def update_learning_rate(self):
#     lrd = self.opt.lr / self.opt.niter_decay
#     lr = self.old_lr - lrd
#     for param_group in self.optimizer_D_A.param_groups:
#         param_group['lr'] = lr
#     for param_group in self.optimizer_D_B.param_groups:
#         param_group['lr'] = lr
#     for param_group in self.optimizer_D_C.param_groups:
#         param_group['lr'] = lr
#     for param_group in self.optimizer_G.param_groups:
#         param_group['lr'] = lr
#
#     print('update learning rate: %f -> %f' % (self.old_lr, lr))
#     self.old_lr = lr

# self.load_network(self.AE, 'AE', which_epoch)
