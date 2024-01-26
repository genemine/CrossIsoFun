from utils import *

def gen_tr_adj_mat(data_tr_list, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_parameter_adaptive_list = []

    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)  # 一个矩阵得出一个阈值，只选择低于这个数字的（1-余弦相似性）元素
        adj_parameter_adaptive_list.append(adj_parameter_adaptive)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, i, adj_metric))

    return adj_train_list, adj_parameter_adaptive_list # [tr_expr_adj, tr_seqdm_adj，tr_PPI_adj]


def gen_te_adj_mat(data_trte_list, trte_idx, adj_parameter_adaptive_list):
    adj_metric = "cosine"  # cosine distance
    adj_test_list = []
    for i in range(len(data_trte_list)):
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive_list[i], i, adj_metric))
    return adj_test_list  # [all_expr_adj(缺tr-tr interaction block, te-te interaction block), all_seqdm_adj, all_PPI_adj](元素为余弦相似度）
