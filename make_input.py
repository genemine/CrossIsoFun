import numpy as np
import pandas as pd

import os



def read_feature():
###expr

isoexpr = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/data/data_set/data_set_first/isoexpr.tsv',
                      sep='\t', header=None)

geneexpr = pd.read_csv('/home/yangch/zs_upload/isoform_function_prediction/data/data_set/data_set_first/geneexpr.tsv',
                       sep='\t', header=None)

num_isoexpr = isoexpr.drop([0, 1], axis=1)

# gene_list= isoexpr.iloc[:,1].drop_duplicates().tolist()

file = r'/home/yangch/zs_upload/isoform_function_prediction/data/IsoFun_data/input/data_set_first/num_gene_map.txt'

num_gene_map = pd.read_csv(file, header=None, index_col=None, sep='\t')

# gene变成num，使用字典和map函数
gene2num_dict = dict(zip(list(num_gene_map.iloc[:, 1]), list(num_gene_map.iloc[:, 0])))
geneexpr.iloc[:, 0] = geneexpr.iloc[:, 0].map(gene2num_dict)

geneexpr = geneexpr.sort_values(by=0)

num_geneexpr = geneexpr.drop([0], axis=1)

all_expr_array = np.vstack((num_isoexpr.values, num_geneexpr.values))





def GO_annotation(num_iso, num_GO, iso_gene):
    if os.path.exists(
            '/home/yangch/zs_upload/isoform_function_prediction/method/MOGONET-main/data_set_first/GO_annotation_matrix.txt'):
        GO_annotation_matrix = pd.read_csv(
            '/home/yangch/zs_upload/isoform_function_prediction/method/MOGONET-main/data_set_first/GO_annotation_matrix.txt',
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

        GO_annotation_matrix.to_csv(
            '/home/yangch/zs_upload/isoform_function_prediction/method/MOGONET-main/data_set_first/GO_annotation_matrix.txt',
            sep='\t', index=False, header=False)

    return GO_annotation_matrix




def read_train_gene():


    train_gene_file= '/home/yangch/zs_upload/isoform_function_prediction/data/group_train_test/data_set_first/num_train_list.txt'
    train_gene_list= pd.read_csv(train_gene_file, header=None, index_col=None, sep='\t').iloc[:,0].tolist()
    train_gene_list.sort()

    return train_gene_list

