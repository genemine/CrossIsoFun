import sys

sys.path.append('/home/yangch/zs_upload/isoform_function_prediction/method/MOGONET-main')

import numpy as np
import pandas as pd

import os

from my_utils import *


def read_feature():
    feature_list = []

    ###################################################expr############################################################

    expr_dict = {}

    isoexpr = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/data/data_set/data_set_first/iso_expr.tsv',
                          sep='\t', header=None)

    geneexpr = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/data/data_set/data_set_first/gene_expr.tsv',
                          sep='\t', header=None)

    num_isoexpr = isoexpr.drop([0, 1], axis=1)


    file = r'/home/yangch/zs_upload/isoform_function_prediction/data/IsoFun_data/input/data_set_first/num_gene_map.txt'
    num_gene_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

    # gene变成num，使用字典和map函数
    gene2num_dict = dict(zip(list(num_gene_map.iloc[:, 1]), list(num_gene_map.iloc[:, 0])))
    geneexpr.iloc[:, 0] = geneexpr.iloc[:, 0].map(gene2num_dict)


    geneexpr = geneexpr.sort_values(by=0)

    num_geneexpr = geneexpr.drop([0], axis=1).apply(lambda x: x / x.max())

    expr_dict['iso'] = num_isoexpr.values
    expr_dict['gene'] = num_geneexpr.values
    # ['iso':iso_expression_matrix, 'gene': gene_expression_matrix]

    ################################################domain##############################################################

    seqdm_dict = {}

    isoseqdm = pd.read_csv(
        '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/iso_seqdm.txt',
        sep='\t', header=None)

    geneseqdm = pd.read_csv(
        '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/gene_seqdm.txt',
        sep='\t', header=None)

    num_isoseqdm = isoseqdm.drop([0, 1], axis=1)

    # gene变成num，使用字典和map函数
    geneseqdm.iloc[:, 0] = geneseqdm.iloc[:, 0].map(gene2num_dict)

    geneseqdm = geneseqdm.sort_values(by=0)

    num_geneseqdm = geneseqdm.drop([0], axis=1)

    seqdm_dict['iso'] = num_isoseqdm.values
    seqdm_dict['gene'] = num_geneseqdm.values
    # ['iso':iso_domain_matrix, 'gene': gene_domain_matrix]

    #####################################################PPI############################################################
    PPI_dict = {}

    isoPPI = pd.read_csv(
        '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/iso_PPI.txt',
        sep='\t', header=None)

    genePPI = pd.read_csv(
        '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/gene_PPI.txt',
        sep='\t', header=None)

    num_isoPPI = isoPPI.drop([0, 1], axis=1)

    # gene变成num，使用字典和map函数
    genePPI.iloc[:, 0] = genePPI.iloc[:, 0].map(gene2num_dict)

    genePPI = genePPI.sort_values(by=0)

    num_genePPI = genePPI.drop([0], axis=1)


    PPI_dict['iso'] = num_isoPPI.values
    PPI_dict['gene'] = num_genePPI.values
    # ['iso':iso_domain_matrix, 'gene': gene_domain_matrix]

    # 添加到list
    feature_list.append(expr_dict)
    feature_list.append(seqdm_dict)
    feature_list.append(PPI_dict)

    feature_name = []
    feature_name.append('expr')
    feature_name.append('seqdm')
    feature_name.append('PPI')

    return feature_list, feature_name


def train_test_iso_gene_partition():

    iso_gene = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/iso_gene.txt',
                          sep='\t', header=None)


    ## 读取gene列表
    file = r'/home/yangch/zs_upload/isoform_function_prediction/data/IsoFun_data/input/data_set_first/num_gene_map.txt'
    num_gene_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

    ## gene变成num，使用字典和map函数
    gene2num_dict = dict(zip(list(num_gene_map.iloc[:, 1]), list(num_gene_map.iloc[:, 0])))
    iso_gene.iloc[:, 1] = iso_gene.iloc[:, 1].map(gene2num_dict)

    ## 读取isoform列表
    file = r'/home/yangch/zs_upload/isoform_function_prediction/data/IsoFun_data/input/data_set_first/num_isoform_map.txt'
    num_isoform_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

    ## isoform变成num，使用字典和map函数
    isoform2num_dict = dict(zip(list(num_isoform_map.iloc[:, 1]), list(num_isoform_map.iloc[:, 0])))
    iso_gene.iloc[:, 0] = iso_gene.iloc[:, 0].map(isoform2num_dict)


    ###training data
    train_gene_list = read_train_gene()
    iso_of_train_gene = gene_list_find_iso_list(iso_gene, train_gene_list)

    num_gene = len(set(iso_gene.iloc[:, 1]))

    train_gene_mask = np.zeros(num_gene)

    for gene in train_gene_list:
        # print(gene)
        train_gene_mask[gene - 1] = 1

    train_gene_mask = train_gene_mask > 0.5  # 0 1变成bool

    sig_flag = get_sig_flag(iso_gene)
    mig_flag = np.array([not x for x in sig_flag])

    train_iso_mask_paired = np.array(sig_flag).copy()
    train_iso_mask_unpaired = np.array(mig_flag).copy()

    for i in range(len(train_iso_mask_paired)):
        if (i + 1) not in iso_of_train_gene:  # isoform属于训练基因且SIG-->paired
            train_iso_mask_paired[i] = False

    for i in range(len(train_iso_mask_unpaired)):
        if (i + 1) not in iso_of_train_gene:  # isoform属于训练基因且MIG-->unpaired
            train_iso_mask_unpaired[i] = False

    ##generate iso_gene_tr

    num_train_gene = np.sum(train_gene_mask)
    num_train_iso_paired = np.sum(train_iso_mask_paired)

    iso_gene_tr_paired = pd.DataFrame(np.zeros([num_train_gene + num_train_iso_paired, 2]))

    iso_gene_tr_paired.iloc[:, 0] = list(range(0, num_train_gene + num_train_iso_paired)) #从0开始
    iso_gene_tr_paired.iloc[:, 1] = list(range(0, num_train_gene + num_train_iso_paired))

    iso_gene_tr_unpaired = iso_gene[train_iso_mask_unpaired]
    iso_gene_tr_unpaired.index = range(len(iso_gene_tr_unpaired))

    iso_gene_tr_unpaired.iloc[:, 0] = list(range(len(iso_gene_tr_paired), len(iso_gene_tr_paired)+len(iso_gene_tr_unpaired)))#从len_paired开始
    list_gene = [len(iso_gene_tr_paired)]
    current_index = len(iso_gene_tr_paired)
    for i in range(1, len(iso_gene_tr_unpaired)):
        if iso_gene_tr_unpaired.iloc[i, 1] != iso_gene_tr_unpaired.iloc[i - 1, 1]:
            current_index += 1
        list_gene.append(current_index)
    iso_gene_tr_unpaired.iloc[:, 1] = list_gene

    # iso_gene_tr_paired = iso_gene[train_iso_mask_paired]
    # iso_gene_tr_paired.index = range(len(iso_gene_tr_paired))
    # iso_gene_tr_unpaired = iso_gene[train_iso_mask_unpaired]
    # iso_gene_tr_unpaired.index = range(len(iso_gene_tr_unpaired))

    # iso_gene_tr_paired.iloc[:, 0] = list(range(0, len(iso_gene_tr_paired)))#从0开始
    # iso_gene_tr_paired.iloc[:, 1] = list(range(0, len(iso_gene_tr_paired)))
    #
    # iso_gene_tr_unpaired.iloc[:,0] = list(range(len(iso_gene_tr_paired), len(iso_gene_tr_paired) + len(iso_gene_tr_unpaired)))
    # list_gene = [len(iso_gene_tr_paired)]
    # current_index = len(iso_gene_tr_paired)
    # for i in range(1,len(iso_gene_tr_unpaired)):
    #     if iso_gene_tr_unpaired.iloc[i, 1] != iso_gene_tr_unpaired.iloc[i-1, 1]:
    #         current_index += 1
    #     list_gene.append(current_index)
    # iso_gene_tr_unpaired.iloc[:, 1] = list_gene


    ###测试

    test_gene_mask = np.array([not x for x in train_gene_mask])

    test_gene_list = read_test_gene()

    iso_of_test_gene = gene_list_find_iso_list(iso_gene, test_gene_list)

    test_iso_mask_paired = np.array(sig_flag).copy()
    test_iso_mask_unpaired = np.array(mig_flag).copy()

    for i in range(len(test_iso_mask_paired)):
        if (i + 1) not in iso_of_test_gene:  # isoform属于测试基因且SIG-->paired
            test_iso_mask_paired[i] = False

    for i in range(len(test_iso_mask_unpaired)):
        if (i + 1) not in iso_of_test_gene:  # isoform属于测试基因且MIG-->unpaired
            test_iso_mask_unpaired[i] = False

    iso_gene_te_paired = iso_gene[test_iso_mask_paired]
    iso_gene_te_paired.index = range(len(iso_gene_te_paired))
    iso_gene_te_unpaired = iso_gene[test_iso_mask_unpaired]
    iso_gene_te_unpaired.index = range(len(iso_gene_te_unpaired))#从0开始

    iso_gene_te_paired.iloc[:, 0] = list(range(len(iso_gene_te_paired)))
    iso_gene_te_paired.iloc[:, 1] = list(range(len(iso_gene_te_paired)))

    iso_gene_te_unpaired.iloc[:,0] = list(range(len(iso_gene_te_paired), len(iso_gene_te_paired) + len(iso_gene_te_unpaired)))#从len_paired开始
    list_gene = [len(iso_gene_te_paired)]
    current_index = len(iso_gene_te_paired)
    for i in range(1,len(iso_gene_te_unpaired)):
        if iso_gene_te_unpaired.iloc[i, 1] != iso_gene_te_unpaired.iloc[i-1, 1]:
            current_index += 1
        list_gene.append(current_index)
    iso_gene_te_unpaired.iloc[:, 1] = list_gene

    iso_gene_trte = {'train':[iso_gene_tr_paired, iso_gene_tr_unpaired], 'test':[iso_gene_te_paired,iso_gene_te_unpaired]}# iso_gene tr和te的index，分paired和unpaired,paired都是从0开始

    # test_iso_mask= np.zeros(len(train_iso_mask))
    #
    #
    # for i in range(len(test_iso_mask)):
    #     if (i+1) in iso_of_test_gene:   #isoform属于测试基因
    #         test_iso_mask[i]= 1
    #
    # test_iso_mask= test_iso_mask.astype(np.bool)

    return train_iso_mask_paired, train_iso_mask_unpaired, test_iso_mask_paired, test_iso_mask_unpaired, \
        train_gene_mask, test_gene_mask, iso_gene_trte


def input_file_feature():
    feature_list, feature_name = read_feature()

    train_iso_mask_paired, train_iso_mask_unpaired, test_iso_mask_paired, test_iso_mask_unpaired,\
        train_gene_mask, _, _ = train_test_iso_gene_partition()

    for i in range(len(feature_list)):
        # print(i)

        out_train_gene = feature_list[i]['gene'][train_gene_mask]

        out_train_iso_paired = feature_list[i]['iso'][train_iso_mask_paired]
        out_train_iso_unpaired = feature_list[i]['iso'][train_iso_mask_unpaired]

        out_test_iso_paired = feature_list[i]['iso'][test_iso_mask_paired]
        out_test_iso_unpaired = feature_list[i]['iso'][test_iso_mask_unpaired]

        # out_test_gene = feature_list[i]['gene'][test_gene_mask]

        out_train_iso_paired = np.vstack((out_train_gene, out_train_iso_paired)) #可去掉
        # out_test= out_test_iso

        out_path = '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/input_file'

        pd.DataFrame(out_train_iso_paired).to_csv(out_path + '/' + feature_name[i] + '_tr_paired.txt', sep='\t',
                                                  index=False, header=False)
        pd.DataFrame(out_train_iso_unpaired).to_csv(out_path + '/' + feature_name[i] + '_tr_unpaired.txt', sep='\t',
                                                    index=False, header=False)
        pd.DataFrame(out_test_iso_paired).to_csv(out_path + '/' + feature_name[i] + '_te_paired.txt', sep='\t',
                                                 index=False, header=False)
        pd.DataFrame(out_test_iso_unpaired).to_csv(out_path + '/' + feature_name[i] + '_te_unpaired.txt', sep='\t',
                                                   index=False, header=False)
    return


def GO_annotation(num_iso, num_GO, iso_gene):
    if os.path.exists(
            '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/GO_annotation_matrix.txt'):
        GO_annotation_matrix = pd.read_csv(
            '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/GO_annotation_matrix.txt',
            sep='\t',
            header=None)
    else:

        GO_annotation_matrix = pd.DataFrame(np.zeros((num_iso, num_GO)))

        file = r'/home/yangch/zs_upload/isoform_function_prediction/data/IsoFun_data/input/data_set_first/num_GO_map.txt'
        num_GO_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

        path = '/home/yangch/zs_upload/isoform_function_prediction/data/data_set/data_set_first/goterms'
        filenames = os.listdir(path)
        for filename in filenames:
            # print(filename)

            GO = filename.split('.')[0]
            GO = GO.replace('_', ':')

            print(GO)

            GO_num = num_GO_map.iloc[list(num_GO_map.iloc[:, 1] == GO), 0]
            # GO_num= GO_num.values

            # print(GO_num.values)

            # break

            genes_of_GO = pd.read_csv(path + r'/' + filename, header=None, index_col=None, sep='\t')

            for row_index in range(iso_gene.shape[0]):
                if iso_gene.iloc[row_index, 1] in genes_of_GO.iloc[:, 0].tolist():
                    GO_annotation_matrix.iloc[row_index, GO_num - 1] = 1

        # GO_annotation_matrix.to_csv(
        #     '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/GO_annotation_matrix_iso.txt',
        #     sep='\t', index=False, header=False)

    return GO_annotation_matrix


def input_file_label():
    # isoexpr = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/data/data_set/data_set_first/isoexpr.tsv', sep='\t', header=None)

    # iso_gene = isoexpr.iloc[:, :2]
    
    iso_gene = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/iso_gene.txt',
                          sep='\t', header=None)

    num_iso = iso_gene.shape[0]
    num_GO = 96

    iso_annotation = GO_annotation(num_iso, num_GO, iso_gene)
    gene_annotation = GO_annotation2gene_label(iso_annotation, iso_gene)
    gene_annotation = gene_annotation.iloc[:, 1:]

    train_iso_mask_paired, train_iso_mask_unpaired, test_iso_mask_paired, test_iso_mask_unpaired,\
        train_gene_mask, test_gene_mask, iso_gene_trte = train_test_iso_gene_partition()

    out_train_gene_label = gene_annotation.iloc[train_gene_mask, :]
    out_train_iso_label_paired = iso_annotation.iloc[train_iso_mask_paired,:]
    out_train_iso_label_unpaired = iso_annotation.iloc[train_iso_mask_unpaired,:]

    out_train_label= pd.concat([out_train_gene_label, out_train_iso_label_paired, out_train_iso_label_unpaired])

    out_test_gene_label = gene_annotation.iloc[test_gene_mask, :]


    # out_train_path= '/home/yangch/zs_upload/isoform_function_prediction/method/MOGONET-main/data_set_first/input_file/train_label_input'

    # for i in range(out_train_label.shape[1]):
    # out_train_label.iloc[:,i].to_csv(out_train_path+ '/'+ str(i)+ '_labels_tr.txt', sep='\t', index=False, header=False)

    # out_test_path= '/home/yangch/zs_upload/isoform_function_prediction/method/MOGONET-main/data_set_first/input_file/test_label_input'

    # out_test_gene_label= gene_annotation.iloc[test_gene_mask,:]
    # for i in range(out_test_gene_label.shape[1]):
    # out_test_gene_label.iloc[:,i].to_csv(out_test_path+ '/'+ str(i)+ '_gene_labels_te.txt', sep='\t', index=False, header=False)

    out_path = '/home/yangch/zs_upload/isoform_function_prediction/method/CrossIsoFun/data_set_first/input_file'
    # out_train_gene_label.to_csv(out_path + '/' + 'gene_labels_tr_lww.txt', sep='\t', index=False, header=False)
    out_train_label.to_csv(out_path + '/' + 'labels_tr_lww.txt', sep='\t', index=False, header=False)
    out_test_gene_label.to_csv(out_path + '/' + 'gene_labels_te_lww.txt', sep='\t', index=False, header=False)

    np.save(out_path+'/iso_gene_trte.npy', iso_gene_trte)
    return