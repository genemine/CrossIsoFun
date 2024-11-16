import os
import numpy as np
import pandas as pd
from my_utils import *
import torch


def create_ordered_binary_matrix(ppi_data, ordered_proteins_list):

    # Map each protein to its index in the ordered list
    protein_to_index = {protein: idx for idx, protein in enumerate(ordered_proteins_list)}

    # Initialize a binary matrix
    matrix_size = len(ordered_proteins_list)
    binary_matrix = np.zeros((matrix_size, matrix_size), dtype=int)


    # Fill the matrix based on the PPI data
    for interaction in ppi_data:
        protein_a, protein_b = interaction.strip().split(', ')
        if protein_a in protein_to_index and protein_b in protein_to_index:
            index_a, index_b = protein_to_index[protein_a], protein_to_index[protein_b]
            binary_matrix[index_a][index_b] = 1
            binary_matrix[index_b][index_a] = 1  # Assuming the interactions are bidirectional

    for i in range(matrix_size):
        binary_matrix[i][i] = 1  # Assuming the interactions are bidirectional

    return binary_matrix




def read_PPI(data_folder, iso_gene_ordered):
    # Create a sorted list of all unique proteins
    row_gene = iso_gene_ordered[1].drop_duplicates(keep='first').tolist()
    protein_list = iso_gene_ordered[1].values

    iso_row_idx = []
    for protein in protein_list:
        iso_row_idx.append(row_gene.index(protein))

    file_path = data_folder + 'iso_PPI_demo.txt'

    # Open and read the contents of the file
    with open(file_path, 'r') as file:
        ppi_data = file.readlines()
            
    # assign interactions
    ppi_matrix = create_ordered_binary_matrix(ppi_data, row_gene)
    print('finish the interaction assignment')

    # Convert to DataFrame
    df_matrix = pd.DataFrame(ppi_matrix)
    print('# of edges in PPI: ' + str(np.sum(np.sum(df_matrix))))

    # Convert the isoform-level PPI
    iso_PPI = df_matrix.iloc[iso_row_idx, :]
    iso_PPI.index = range(len(iso_PPI))
    return iso_PPI

def read_feature(train_data_folder, test_data_folder, data_folder, iso_gene):
    gene_in_order = iso_gene[1].drop_duplicates(keep='first').tolist()
    iso_gene_new_idx = []
    for gene in gene_in_order:
        idx = np.where(iso_gene[1] == gene)[0].tolist()
        iso_gene_new_idx += idx
    iso_gene_ordered = iso_gene.iloc[iso_gene_new_idx, :]
    iso_gene_ordered.index = range(len(iso_gene_ordered))

    ###################################################expr############################################################
    expr_dict = {}

    isoexpr_train = pd.read_csv(train_data_folder + '/iso_expr.txt', sep='\t', header=None)
    isoexpr_train.index = range(len(isoexpr_train))
    num_isoexpr_train = isoexpr_train.drop([0,1], axis=1).values

    isoexpr_test = pd.read_csv(test_data_folder + '/iso_expr.txt', sep='\t', header=None)
    isoexpr_test.index = range(len(isoexpr_test))
    num_isoexpr_test = isoexpr_train.drop([0,1], axis=1).values

    num_isoexpr = np.vstack([num_isoexpr_train, num_isoexpr_test])
    num_isoexpr = num_isoexpr / np.max(abs(num_isoexpr))
    num_isoexpr = num_isoexpr[iso_gene_new_idx, :]

    expr_dict['iso'] = num_isoexpr
    ###################################################seqdm############################################################
    seqdm_dict = {}

    isoseqdm_train = pd.read_csv(train_data_folder + '/iso_seqdm.txt', sep='\t', header=None)
    isoseqdm_train.index = range(len(isoseqdm_train))
    num_isoseqdm_train = isoseqdm_train.drop([0,1], axis=1).values

    isoseqdm_test = pd.read_csv(test_data_folder + '/iso_seqdm.txt', sep='\t', header=None)
    isoseqdm_test.index = range(len(isoseqdm_test))
    num_isoseqdm_test = isoseqdm_test.drop([0,1], axis=1).values

    num_isoseqdm = np.vstack([num_isoseqdm_train, num_isoseqdm_test])
    num_isoseqdm = num_isoseqdm / np.max(abs(num_isoseqdm), axis=0)
    num_isoseqdm = num_isoseqdm[iso_gene_new_idx, :]

    seqdm_dict['iso'] = num_isoseqdm
    ###################################################PPI############################################################
    PPI_dict = {}

    isoPPI = read_PPI(data_folder, iso_gene_ordered)
    num_isoPPI = isoPPI.values

    PPI_dict['iso'] = num_isoPPI
    ####################################################################################################################

    # 添加到list
    feature_list = []
    feature_list.append(expr_dict)
    feature_list.append(seqdm_dict)
    feature_list.append(PPI_dict)

    feature_name = []
    feature_name.append('expr')
    feature_name.append('seqdm')
    feature_name.append('PPI')

    return feature_list, feature_name, iso_gene_ordered


def train_test_partition(train_data_folder, test_data_folder, iso_gene):
    ###training data

    iso_of_train_gene = read_train_isoform(train_data_folder, iso_gene)

    sig_flag = get_sig_flag(iso_gene)
    mig_flag = np.array([not x for x in sig_flag])

    train_iso_mask_paired = np.array(sig_flag).copy()
    train_iso_mask_unpaired = np.array(mig_flag).copy()

    for i in range(len(train_iso_mask_paired)):
        if iso_gene.iloc[i, 0] not in iso_of_train_gene:  # isoform属于训练基因且SIG-->paired
            train_iso_mask_paired[i] = False

    for i in range(len(train_iso_mask_unpaired)):
        if iso_gene.iloc[i, 0] not in iso_of_train_gene:  # isoform属于训练基因且MIG-->unpaired
            train_iso_mask_unpaired[i] = False

    ###testing data

    iso_of_test_gene = read_test_isoform(test_data_folder, iso_gene)

    test_iso_mask_paired = np.array(sig_flag).copy()
    test_iso_mask_unpaired = np.array(mig_flag).copy()

    for i in range(len(test_iso_mask_paired)):
        if iso_gene.iloc[i, 0] not in iso_of_test_gene:  # isoform属于测试基因且SIG-->paired
            test_iso_mask_paired[i] = False

    for i in range(len(test_iso_mask_unpaired)):
        if iso_gene.iloc[i, 0] not in iso_of_test_gene:  # isoform属于测试基因且MIG-->unpaired
            test_iso_mask_unpaired[i] = False

    ###generate iso_gene_tr

    num_train_iso_paired = np.sum(train_iso_mask_paired)
    print('# of train and paired isoforms: ' + str(num_train_iso_paired))

    num_train_iso_unpaired = np.sum(train_iso_mask_unpaired)
    print('# of train and unpaired isoforms: ' + str(num_train_iso_unpaired))

    iso_gene_tr_paired = pd.DataFrame(np.zeros([num_train_iso_paired, 2]))
    iso_gene_tr_paired[0] = list(range(0, num_train_iso_paired))  # 从0开始
    iso_gene_tr_paired[1] = list(range(0, num_train_iso_paired))

    train_unpaired_data = iso_gene[train_iso_mask_unpaired]
    iso_gene_tr_unpaired = pd.DataFrame(np.zeros([num_train_iso_unpaired, 2]))
    iso_gene_tr_unpaired[0] = list(
        range(num_train_iso_paired, num_train_iso_paired + num_train_iso_unpaired))  # 从len_paired开始

    list_gene = [len(iso_gene_tr_paired)]
    current_index = len(iso_gene_tr_paired)
    for i in range(1, num_train_iso_unpaired):
        if train_unpaired_data.iloc[i, 1] != train_unpaired_data.iloc[i - 1, 1]:
            current_index += 1
        list_gene.append(current_index)
    iso_gene_tr_unpaired[1] = list_gene

    ##generate iso_gene_te

    num_test_iso_paired = np.sum(test_iso_mask_paired)
    print('# of test and paired isoforms: ' + str(num_test_iso_paired))

    num_test_iso_unpaired = np.sum(test_iso_mask_unpaired)
    print('# of test and unpaired isoforms: ' + str(num_test_iso_unpaired))

    iso_gene_te_paired = pd.DataFrame(np.zeros([num_test_iso_paired, 2]))
    iso_gene_te_paired[0] = list(range(0, num_test_iso_paired))  # 从0开始
    iso_gene_te_paired[1] = list(range(0, num_test_iso_paired))

    test_unpaired_data = iso_gene[test_iso_mask_unpaired]
    iso_gene_te_unpaired = pd.DataFrame(np.zeros([num_test_iso_unpaired, 2]))
    iso_gene_te_unpaired[0] = list(
        range(num_test_iso_paired, num_test_iso_paired + num_test_iso_unpaired))  # 从len_paired开始

    list_gene = [len(iso_gene_te_paired)]
    current_index = len(iso_gene_te_paired)
    for i in range(1, num_test_iso_unpaired):
        if test_unpaired_data.iloc[i, 1] != test_unpaired_data.iloc[i - 1, 1]:
            current_index += 1
        list_gene.append(current_index)
    iso_gene_te_unpaired[1] = list_gene

    iso_gene_trte = {'train': [iso_gene_tr_paired, iso_gene_tr_unpaired], 'test': [iso_gene_te_paired,
                                                                                   iso_gene_te_unpaired]}  # iso_gene tr和te的index，分paired和unpaired,paired都是从0开始

    return train_iso_mask_paired, train_iso_mask_unpaired, test_iso_mask_paired, test_iso_mask_unpaired, iso_gene_trte


def GO_annotation(data_folder, num_iso, num_GO, iso_gene):

    GO_annotation_matrix = pd.DataFrame(np.zeros((num_iso, num_GO)))

    file = data_folder + 'num_GO_map.txt'
    num_GO_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

    path = data_folder + 'goterms'
    filenames = os.listdir(path)
    for filename in filenames:
        # print(filename)

        GO = filename.split('.')[0]
        GO = GO.replace('_', ':')

        GO_num = num_GO_map.iloc[list(num_GO_map.iloc[:, 1] == GO), 0]
        genes_of_GO = pd.read_csv(path + r'/' + filename, header=None, index_col=None, sep='\t')

        for row_index in range(iso_gene.shape[0]):
            if iso_gene.iloc[row_index, 1] in genes_of_GO.iloc[:, 0].tolist():
                GO_annotation_matrix.iloc[row_index, GO_num - 1] = 1

    return GO_annotation_matrix


def prepare_trte_data(train_data_folder, test_data_folder, train_label_folder, data_folder):
    iso_gene_tr = pd.read_csv(train_data_folder + 'iso_gene.txt', sep='\t', header=None)
    iso_gene_te = pd.read_csv(test_data_folder + 'iso_gene.txt', sep='\t', header=None)
    iso_gene = pd.concat([iso_gene_tr, iso_gene_te], axis=0)
    iso_gene.index = range(len(iso_gene))
    print(iso_gene)

    ### features
    feature_list, feature_name, iso_gene = read_feature(train_data_folder, train_data_folder, data_folder, iso_gene)
    num_feature = len(feature_list)

    train_iso_mask_paired, train_iso_mask_unpaired, test_iso_mask_paired, test_iso_mask_unpaired, iso_gene_trte \
        = train_test_partition(train_data_folder, test_data_folder, iso_gene)

    data_tr_list_paired = []  # [tr_expr_matrix,tr_seqdm_matrix,tr_PPI_matrix]
    data_tr_list_unpaired = []  # [tr_expr_matrix_unpaired,tr_seqdm_matrix_unpaired,tr_PPI_matrix_unpaired]
    data_te_list_paired = []  # [te_expr_matrix,te_segdm_matrix,te_PPI_matrix]
    data_te_list_unpaired = []

    for i in range(num_feature):
        data_tr_list_paired.append(feature_list[i]['iso'][train_iso_mask_paired])
        data_tr_list_unpaired.append(feature_list[i]['iso'][train_iso_mask_unpaired])
        data_te_list_paired.append(feature_list[i]['iso'][test_iso_mask_paired])
        data_te_list_unpaired.append(feature_list[i]['iso'][test_iso_mask_unpaired])

    iso2gene_te = np.vstack([iso_gene.values[test_iso_mask_paired],iso_gene.values[test_iso_mask_unpaired]])

    num_tr_paired = data_tr_list_paired[0].shape[0]  # 训练paired样本数量
    num_tr_unpaired = data_tr_list_unpaired[0].shape[0]  # 训练unpaired样本数量
    num_te_paired = data_te_list_paired[0].shape[0]  # 测试paired样本数量
    num_te_unpaired = data_te_list_unpaired[0].shape[0]  # 测试unpaired样本数量

    data_mat_list = []  # [all_expr_matrix,all_seqdm_matrix,all_PPI_matrix]
    for i in range(num_feature):
        data_mat_list.append(np.concatenate(
            (data_tr_list_paired[i], data_tr_list_unpaired[i], data_te_list_paired[i], data_te_list_unpaired[i]),
            axis=0))
    data_tensor_list = []  # [tensor(all_expr_matrix),tensor(all_domain_matrix)]
    for i in range(num_feature):
        data_tensor_list.append(torch.tensor(data_mat_list[i], dtype=torch.float32))

    iso_idx_dict = {}  # {tr:1,2,… ;te:23816,23817,…}
    iso_idx_dict["tr_paired"] = list(range(num_tr_paired))
    iso_idx_dict["tr_unpaired"] = list(range(num_tr_paired, num_tr_paired + num_tr_unpaired))
    iso_idx_dict["te_paired"] = list(
        range(num_tr_paired + num_tr_unpaired, num_tr_paired + num_tr_unpaired + num_te_paired))
    iso_idx_dict["te_unpaired"] = list(range(num_tr_paired + num_tr_unpaired + num_te_paired,
                                             num_tr_paired + num_tr_unpaired + num_te_paired + num_te_unpaired))

    data_train_list = []
    data_all_list = []

    for i in range(num_feature):
        data_train_list.append(
            torch.cat((data_tensor_list[i][iso_idx_dict["tr_paired"]].clone(),
                       data_tensor_list[i][iso_idx_dict["tr_unpaired"]].clone()),
                      0))  # [tensor(tr_expr_matrix),tensor(tr_seqdm_matrix), tensor(tr_PPI_matrix]
        data_all_list.append(torch.cat((data_tensor_list[i][iso_idx_dict["tr_paired"]].clone(),
                                        data_tensor_list[i][iso_idx_dict["tr_unpaired"]].clone(),
                                        data_tensor_list[i][iso_idx_dict["te_paired"]].clone(),
                                        data_tensor_list[i][iso_idx_dict["te_unpaired"]].clone()),
                                       0))  # [tensor(all_expr_matrix),tensor(all_domain_matrix)]

    print('feature shape: ' + str(data_all_list[0].shape) + str(data_all_list[1].shape) + str(data_all_list[2].shape))

    num_iso = iso_gene.shape[0]
    file = train_label_folder + 'num_GO_map.txt'
    GO_list = pd.read_csv(file, header=None, index_col=None, sep='\t')[1].tolist()
    num_GO = len(GO_list)

    # labels
    iso_annotation = GO_annotation(train_label_folder, num_iso, num_GO, iso_gene)

    out_train_iso_label_paired = iso_annotation[train_iso_mask_paired]
    out_train_iso_label_unpaired = iso_annotation[train_iso_mask_unpaired]

    out_train_label = pd.concat([out_train_iso_label_paired, out_train_iso_label_unpaired], axis=0)
    out_train_label.index = range(len(out_train_label))
    print('train isoform label shape: ' + str(out_train_label.shape))

    iso_gene_train = pd.concat([iso_gene_trte['train'][0], iso_gene_trte['train'][1]])
    iso_gene_train.index = range(len(iso_gene_train))

    out_train_gene_label = GO_annotation2gene_label(out_train_label, iso_gene_train).iloc[:, 1:].values
    print('train gene label shape: ' + str(out_train_gene_label.shape))


    out_test_iso_label_paired = iso_annotation[test_iso_mask_paired]
    out_test_iso_label_unpaired = iso_annotation[test_iso_mask_unpaired]

    out_test_label = pd.concat([out_test_iso_label_paired, out_test_iso_label_unpaired])
    out_test_label.index = range(len(out_test_label))
    print('test isoform label shape: ' + str(out_test_label.shape))

    iso_gene_test = pd.DataFrame(pd.concat([iso_gene_trte['test'][0], iso_gene_trte['test'][1]]).values)
    iso_gene_test.index = range(len(iso_gene_test))

    out_test_gene_label = GO_annotation2gene_label(out_test_label, iso_gene_test).iloc[:, 1:].values
    print('test gene label shape: ' + str(out_test_gene_label.shape))

    labels_tr = out_train_gene_label.astype(int)
    print('# of postive samples in training set: ' + str(np.sum(labels_tr, axis=0).tolist()))
    labels_te = out_test_gene_label.astype(int)
    print('# of postive samples in testing set: ' + str(np.sum(labels_te, axis=0).tolist()))

    labels = np.concatenate((labels_tr, labels_te))  # 所有样本基因标签矩阵
    print('label shape: ' + str(labels.shape))

    labels_idx_dict = {}  # {tr:1,2,… ;te:23816,23817,…}


    labels_idx_dict['tr'] = list(range(len(labels_tr)))
    labels_idx_dict['te'] = list(range(len(labels_tr), len(labels_tr) + len(labels_te)))

    ###iso_gene ->tr/te order
    idx_tr_paired = list(np.where(train_iso_mask_paired == True)[0])
    idx_tr_unpaired = list(np.where(train_iso_mask_unpaired == True)[0])
    idx_te_paired = list(np.where(test_iso_mask_paired == True)[0])
    idx_te_unpaired = list(np.where(test_iso_mask_unpaired == True)[0])
    new_order = torch.tensor(np.array(idx_tr_paired + idx_tr_unpaired + idx_te_paired + idx_te_unpaired))

    ###iso2gene for integrate III result to PPI
    iso2gene_gene = []
    iso2gene = []
    idx = -1
    for gene in iso_gene.iloc[:, 1].values:
        if gene not in iso2gene_gene:
            idx += 1
            iso2gene_gene.append(gene)
        iso2gene.append(idx)
    iso2gene = torch.tensor(np.array(iso2gene)).to(torch.int64)

    return data_train_list, data_all_list, iso_idx_dict, labels, labels_idx_dict, iso_gene_trte, iso2gene, new_order,GO_list, pd.DataFrame(iso2gene_te)
