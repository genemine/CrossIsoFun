import numpy as np
import pandas as pd


def read_train_isoform(data_folder, iso_gene):

    train_isoform_file = data_folder + 'train_isoform_list.txt'
    train_isoform_list = pd.read_csv(train_isoform_file, header=None, sep='\t').iloc[:,0].tolist()
    isoform_list=[]
    for i in range(iso_gene.shape[0]):
        if iso_gene.iloc[i,0] in train_isoform_list:
            isoform_list.append(iso_gene.iloc[i,0])
    print('# of train isoforms:' + str(len(isoform_list)))
    return isoform_list


def read_test_isoform(data_folder, iso_gene):

    test_isoform_file =  data_folder + 'test_isoform_list.txt'
    test_isoform_list = pd.read_csv(test_isoform_file, header=None, sep='\t').iloc[:,0].tolist()
    isoform_list=[]
    for i in range(iso_gene.shape[0]):
        if iso_gene.iloc[i,0] in test_isoform_list:
            isoform_list.append(iso_gene.iloc[i,0])
    print('# of test isoforms:' + str(len(isoform_list)))
    return isoform_list


def GO_annotation2gene_label(GO_annotation_matrix,iso_gene):
    temp= pd.concat([iso_gene, GO_annotation_matrix],axis=1)
    temp= temp.iloc[:,1:]

    #temp.T.reset_index(drop=True).T

    temp= temp.drop_duplicates(keep='first')

    return temp



def get_sig_flag(iso_gene):
    sig_flag = []
    gene_count = pd.value_counts(list(iso_gene.iloc[:, 1]))

    for i in range(iso_gene.shape[0]):
        if gene_count[iso_gene.iloc[i, 1]] == 1:
            sig_flag.append(True)
        else:
            sig_flag.append(False)

    return sig_flag





def gene_list_find_iso_list(iso_gene,gene_list):

    iso_list= []
    for i in range(iso_gene.shape[0]):
        if iso_gene.iloc[i,1] in gene_list:
            iso_list.append(iso_gene.iloc[i,0])

    return iso_list

def gene_list_find_iso_list_tissue(iso_gene,gene_list):

    iso_list= []
    for gene in gene_list:
        idx = np.where(iso_gene[1] == gene)[0].tolist()
        iso_list+=idx

    return iso_list



def get_iso_gene():
    isoexpr = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/data/data_set/data_set_first/isoexpr.tsv',
                          sep='\t',
                          header=None)

    iso_gene = isoexpr.iloc[:, :2]

    # 读取gene列表
    file = r'/home/yangch/zs_upload/isoform_function_prediction/data/IsoFun_data/input/data_set_first/num_gene_map.txt'
    num_gene_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

    # gene变成num，使用字典和map函数
    gene2num_dict = dict(zip(list(num_gene_map.iloc[:, 1]), list(num_gene_map.iloc[:, 0])))
    iso_gene.iloc[:, 1] = iso_gene.iloc[:, 1].map(gene2num_dict)

    # 读取isoform列表
    file = r'/home/yangch/zs_upload/isoform_function_prediction/data/IsoFun_data/input/data_set_first/num_isoform_map.txt'
    num_isoform_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

    # isoform变成num，使用字典和map函数
    isoform2num_dict = dict(zip(list(num_isoform_map.iloc[:, 1]), list(num_isoform_map.iloc[:, 0])))
    iso_gene.iloc[:, 0] = iso_gene.iloc[:, 0].map(isoform2num_dict)


    return iso_gene

def get_test_iso_mask(iso_gene):


    test_gene_list= read_test_gene()

    iso_of_test_gene= gene_list_find_iso_list(iso_gene, test_gene_list)

    test_iso_mask= np.zeros(iso_gene.shape[0])


    for i in range(len(test_iso_mask)):
        if (i+1) in iso_of_test_gene:   #isoform属于测试基因
            test_iso_mask[i]= 1

    test_iso_mask= test_iso_mask.astype(np.bool)


    return test_iso_mask


def iso_score2genes_score(iso_score, iso_gene):

    gene_score_dict= {}
    for i in range(iso_score.shape[0]):
        if iso_gene.iloc[i,1] in gene_score_dict.keys():
            gene_score_dict[iso_gene.iloc[i,1]].append(iso_score[i])
        else:
            gene_score_dict[iso_gene.iloc[i,1]]= []
            gene_score_dict[iso_gene.iloc[i,1]].append(iso_score[i])



    score_list= []

    for key in gene_score_dict.keys():
        #print(key)
        score_list.append( max(gene_score_dict[key]))

    gene_score_df= pd.DataFrame({0: list(gene_score_dict.keys()),1: score_list})


    return gene_score_df

