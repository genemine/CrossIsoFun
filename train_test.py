""" Training and testing of the model
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from options.train_options import TrainOptions
import torch
import torch.nn.functional as F
from utils import *
from my_utils import *
from create_models import *
import time
from cal_auc import *
from generate_adj_matrix import *
from tqdm import tqdm
from read_data import *

opt = TrainOptions().parse()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def init_loss_dict(loss_type_list):
    loss_dict_init = {}
    for loss_type in loss_type_list:
        loss_dict_init[loss_type] = []
    return loss_dict_init

def train_test(data_folder):

    test_inverval = 5
    test_inverval_pretrain = 5

    data_tr_list, data_trte_list, trte_pair_unpair_idx, labels_trte, labels_trte_idx, iso_gene_trte, iso2gene, new_order, GO_list \
        = prepare_trte_data(data_folder)

    num_GOterm = opt.num_GO

    dim_hvcdn = opt.dim_hvcdn  # pow(num_label, num_feature)
    adj_parameter = opt.adj_parameter
    dim_he_list = [opt.GCN1, opt.GCN2, opt.GCN3]

    num_feature = len(data_tr_list)
    iso_gene_tr = iso_gene_trte['train']
    unpair_iso_gene_tr = iso_gene_tr[1]

    num_label = labels_trte.shape[1]
    dim_list = [x.shape[1] for x in data_tr_list]  # [expr_feature_dim, domain_feature_dim， PPI_dim]
    dim_list.append(len(data_trte_list[0]))
    print(dim_list)

    model = create_model(opt, num_feature, num_label, dim_list, dim_he_list, dim_hvcdn, device)
    torch.cuda.synchronize()
    model.calculate_class_weights(data_tr_list[2].to(torch.int).flatten())

    ############################################Training and Testing####################################################

    start = time.time()
    print('Start training')

    #################################################
    # Step1: CycleGAN
    #################################################
    print("\nPretrain cycleGAN...")
    num_paired_data = len(trte_pair_unpair_idx["tr_paired"])
    num_gene = len(labels_trte_idx['tr'])

    if opt.continue_train == False:

        pre_epoch_cycle = 50
        loss_type_list = ['AE', 'G_C', 'D_C', 'PPI', 'PPI_fake', 'PPI_A', 'PPI_B', 'cycle', 'ALL']

        epoch_train_loss = init_loss_dict(loss_type_list)

        for epoch in range(1, pre_epoch_cycle + 1):
            epoch_start_time = time.time()

            # train
            error_cycle_pretrain_train = init_loss_dict(loss_type_list)
            for i in tqdm(range(num_gene)):

                if i < num_paired_data:
                    data_a = data_tr_list[0][i].view(1, 1, dim_list[0])
                    data_b = data_tr_list[1][i].view(1, 1, dim_list[1])
                    data_c = data_tr_list[2][i].view(1, 1, dim_list[2])
                    model.set_input(data_a, data_b, data_c, iso2gene)
                    model.optimize_pretrain_cycleGAN()
                    errors = model.get_current_errors_cycle_pre()
                    for loss_type in errors:
                        error_cycle_pretrain_train[loss_type] += [errors[loss_type]]

                if num_paired_data < i < num_gene:
                    iso_idx_temp = unpair_iso_gene_tr.iloc[list(unpair_iso_gene_tr.iloc[:, 1] == i), 0].values
                    init_iso_of_gene = True
                    for idx in iso_idx_temp:
                        # j = np.random.randint(0, num_paired_data, 1)
                        # j = j[0]
                        data_a = data_tr_list[0][idx].view(1, 1, dim_list[0])
                        data_b = data_tr_list[1][idx].view(1, 1, dim_list[1])
                        data_c = data_tr_list[2][idx].view(1, 1, dim_list[2])
                        model.set_input(data_a, data_b, data_c, iso2gene)
                        model.optimize_pretrain_cycleGAN_unpaired(init_iso_of_gene)
                        init_iso_of_gene = False
                    gene_PPI = data_tr_list[2][iso_idx_temp[-1]].view(1, 1, dim_list[2])
                    model.optimize_pretrain_cycleGAN_unpaired_batch(gene_PPI)
                    errors = model.get_current_errors_cycle_pre_unpaired()
                    for loss_type in errors:
                        error_cycle_pretrain_train[loss_type] += [errors[loss_type]]

            for loss_type in error_cycle_pretrain_train:
                epoch_train_loss[loss_type].append(round(np.mean(error_cycle_pretrain_train[loss_type]), 5))

            print('\n')
            print('train loss for the above epochs')
            for loss_type in epoch_train_loss:
                print(loss_type + ': ' + str(epoch_train_loss[loss_type]))

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, pre_epoch_cycle, time.time() - epoch_start_time))
            # model.save(epoch)
    print('cycleGAN pretrain is finished')

    #################################################
    # Step2: III generation + GCN
    #################################################
    print("\nPretrain CycleGAN + GCN...")

    iso_gene_tr = iso_gene_trte['train']

    iso_gene_tr_df = pd.concat([iso_gene_trte['train'][0], iso_gene_trte['train'][1]], axis=0)
    iso_gene_tr_tensor = torch.tensor(iso_gene_tr_df.values)

    iso_gene_te_df = pd.concat([iso_gene_trte['test'][0], iso_gene_trte['test'][1]], axis=0)
    iso_gene_te_df[0] = iso_gene_te_df[0].apply(lambda x: x + iso_gene_tr_df.iloc[-1, 0] + 1)
    iso_gene_te_df[1] = iso_gene_te_df[1].apply(lambda x: x + iso_gene_tr_df.iloc[-1, 1] + 1)
    iso_gene_te_tensor = torch.tensor(iso_gene_te_df.values)

    iso_gene_te = [iso_gene_te_df.iloc[0:len(iso_gene_trte['test'][0]), :],
                   iso_gene_te_df.iloc[len(iso_gene_trte['test'][0]):, :]]

    trte_idx = {'tr': trte_pair_unpair_idx["tr_paired"] + trte_pair_unpair_idx["tr_unpaired"],
                'te': trte_pair_unpair_idx["te_paired"] + trte_pair_unpair_idx["te_unpaired"]}

    ####################generat III data##############################

    data_tr_index = list(iso_gene_tr[0].values[:, 0]) + list(iso_gene_tr[1].values[:, 0])
    data_te_index = list(iso_gene_te[0].values[:, 0]) + list(iso_gene_te[1].values[:, 0])
    data_index = data_tr_index + data_te_index

    data_trte_III = []
    for i in tqdm(data_index):
        data_a = data_trte_list[0][i].view(1, 1, dim_list[0])
        data_b = data_trte_list[1][i].view(1, 1, dim_list[1])
        data_c = data_trte_list[2][i].view(1, 1, dim_list[2])
        model.set_input(data_a, data_b, data_c, iso2gene)
        dataset_fakec = model.test_unpaired_c().view(1, dim_list[3])
        data_trte_III.append(dataset_fakec)
    data_trte_III = torch.cat(data_trte_III, dim=0).detach().cpu()

    ################symmetrize III data#############################
    data_trte_III = torch.index_select(data_trte_III, 1, new_order)

    data_trte_III_T = data_trte_III.transpose(0, 1)
    mask_1_T = data_trte_III_T >= 0.5
    data_trte_III_T_1 = torch.zeros(data_trte_III_T.shape)
    data_trte_III_T_1[mask_1_T] = data_trte_III_T[mask_1_T]
    mask_1 = data_trte_III_T_1 > data_trte_III
    data_trte_III[mask_1] = data_trte_III[mask_1] + data_trte_III_T_1[mask_1] - data_trte_III[mask_1]

    data_trte_III_copy = data_trte_III
    mask = data_trte_III_copy < 0.5
    data_trte_III_copy[mask] = (data_trte_III[mask] + data_trte_III_T[mask])/2
    data_trte_III = data_trte_III_copy

    data_tr_III = data_trte_III[0:len(data_tr_index), :]

    data_trte_list[2] = data_trte_III
    data_tr_list[2] = data_tr_III

    ################GCN training#############################

    labels_tr_tensor = torch.tensor(labels_trte[labels_trte_idx["tr"]], dtype=torch.float32)

    sample_weight_tr = np.zeros([len(labels_trte_idx["tr"]),
                                 num_label])  # tensor: (N_trsample, 96)  label weight matrix: 若样本标签为0则值为count(0)/(count(0)+count(1)), 为1则为count(1)/(count(0)+count(1))
    for GO in range(labels_trte.shape[1]):
        sample_weight_tr[:, GO] = cal_sample_weight(labels_trte[labels_trte_idx["tr"], GO], num_GOterm)
    sample_weight_tr = torch.tensor(sample_weight_tr)

    adj_tr_list, adj_parameter_adaptive_list =  gen_tr_adj_mat(data_tr_list, adj_parameter)
    adj_trte_list = gen_te_adj_mat(data_trte_list, trte_idx, adj_parameter_adaptive_list)

    # start train
    epoch_train_cycleGAN_GCN = 120

    iso_gene_tr_dict = {}
    iso_gene_tr_dict[0] = torch.tensor(iso_gene_tr[0].values)
    iso_gene_tr_dict[1] = torch.tensor(iso_gene_tr[1].values)

    iso_gene_te_dict = {}
    iso_gene_te_dict[0] = torch.tensor(iso_gene_te[0].values)
    iso_gene_te_dict[1] = torch.tensor(iso_gene_te[1].values)

    loss_type_list = ['GCN_A', 'GCN_B', 'GCN_C', 'GCN_all']

    epoch_train_loss = init_loss_dict(loss_type_list)

    if opt.continue_train_after_GCN == False:

        for epoch in range(1, epoch_train_cycleGAN_GCN + 1):

            epoch_start_time = time.time()
            model.set_input_AE_GCN(data_tr_list, adj_tr_list, labels_tr_tensor, sample_weight_tr, iso_gene_tr_dict)
            model.optimize_cycleGAN_GCN_all()

            errors = model.get_current_errors_cycleGAN_GCN()
            for loss_type in errors:
                epoch_train_loss[loss_type].append(round(errors[loss_type], 5))

            print('\n')
            print('train loss for the above epochs')
            for loss_type in epoch_train_loss:
                print(loss_type + ': ' + str(epoch_train_loss[loss_type]))
            print('\n')

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, epoch_train_cycleGAN_GCN, time.time() - epoch_start_time))
            print('\n')

        # model.save_GCN()
    print('cycleGAN + GCN training is finished')


    #################################################
    # Step4: GCN + VCDN
    #################################################
    print("\n Train GCN + VCDN...")
    epoch_train_GCN_VCDN = 150

    for epoch in range(1, epoch_train_GCN_VCDN + 1):

        print('start training VCDN')
        epoch_start_time = time.time()

        model.set_input_GCN_VCDN(data_tr_list, adj_tr_list, labels_tr_tensor, sample_weight_tr, iso_gene_tr_dict)
        model.optimize_GCN_VCDN()

        # plot loss curve
        errors = model.get_current_errors_GCN_VCDN()
        print(errors)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, epoch_train_GCN_VCDN, time.time() - epoch_start_time))

        ##Testing Phase
        if epoch % test_inverval == 0:
            with torch.no_grad():
                print('testing')

                model.set_input_GCN_VCDN(data_trte_list, adj_trte_list, labels_tr_tensor, sample_weight_tr,
                                         iso_gene_tr_dict)
                pred_result, pred_prob = model.test_GCN_VCDN()
                pred_prob = pred_prob.data.cpu().numpy()

                te_prob = []
                for gene_idx in range(iso_gene_te_tensor[0, 1], iso_gene_te_tensor[-1, 1] + 1):
                    iso_idx = iso_gene_te_tensor[torch.where(iso_gene_te_tensor[:, 1] == gene_idx)[0], 0].numpy()
                    te_prob.append(np.max(pred_prob[iso_idx, :], 0))
                te_prob = np.vstack(te_prob)

                tr_prob = []
                for gene_idx in range(iso_gene_tr_tensor[0, 1], iso_gene_tr_tensor[-1, 1] + 1):
                    iso_idx = iso_gene_tr_tensor[torch.where(iso_gene_tr_tensor[:, 1] == gene_idx)[0], 0].numpy()
                    tr_prob.append(np.max(pred_prob[iso_idx, :], 0))
                tr_prob = np.vstack(tr_prob)

                test_aucs = []
                test_auprcs = []
                test_sig_aucs = []
                test_sig_auprcs = []
                train_aucs = []

                for train_GO_index in range(labels_trte.shape[1]):

                    # print(train_GO_index)

                    labels_tr_GO = labels_trte[labels_trte_idx["tr"], train_GO_index]
                    labels_te_GO = labels_trte[labels_trte_idx["te"], train_GO_index]
                    labels_te_GO_sig = labels_te_GO[0:len(iso_gene_trte['test'][0])]

                    if len(set(labels_te_GO)) == 2:
                        test_auc = roc_auc_score(labels_te_GO, te_prob[:, train_GO_index])
                        test_aucs.append(test_auc)

                        # print("Test AUC: {:.6f}".format(test_auc))
                        test_auprc = baseline_auprc(list(labels_te_GO), list(te_prob[:, train_GO_index]), 0.1)
                        test_auprcs.append(test_auprc)
                        # print("Test AUPRC: {:.6f}".format(test_auprc))
                    else:
                        test_auc = -1
                        test_auprc = -1

                    if len(set(labels_te_GO_sig)) == 2:
                        test_sig_auc = roc_auc_score(labels_te_GO_sig,
                                                     te_prob[:, train_GO_index][0:len(iso_gene_trte['test'][0])])
                        test_sig_aucs.append(test_sig_auc)

                        # print("Test SIG-level AUC: {:.6f}".format(test_sig_auc))
                        test_sig_auprc = baseline_auprc(list(labels_te_GO_sig), list(
                            te_prob[:, train_GO_index][0:len(iso_gene_trte['test'][0])]), 0.1)
                        test_sig_auprcs.append(test_sig_auprc)

                        # print("Test SIG-level AUPRC: {:.6f}".format(test_sig_auprc))
                    else:
                        test_sig_auc = -1
                        test_sig_auprc = -1

                    if len(set(labels_tr_GO)) == 2:
                        train_auc = roc_auc_score(labels_tr_GO, tr_prob[:, train_GO_index])
                        train_aucs.append(train_auc)
                        # print("Train AUC: {:.6f}".format(train_auc))
                    else:
                        train_auc = -1

                print("Test AUC: {:.6f}".format(np.median(test_aucs)))
                print("Test AUPRC: {:.6f}".format(np.median(test_auprcs)))
                print("SIG-level Test AUPRC: {:.6f}".format(np.median(test_sig_auprcs)))
                print("SIG-level Test AUC: {:.6f}".format(np.median(test_sig_aucs)))
                print("Train AUC: {:.6f}".format(np.median(train_aucs)))
                print("\nTest: Epoch {:d}".format(epoch))


    print('GCN + VCDN training is finished')

    print('End of training phase \t Time Taken: %d sec' %
          (time.time() - start))
