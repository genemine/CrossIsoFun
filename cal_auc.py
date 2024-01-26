import numpy as np
import pandas as pd
from sklearn import metrics
import os
import random


def cal_auc(label_list, score_list):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(label_list), np.array(score_list))
    auc = metrics.auc(fpr, tpr)

    ret = auc

    return (ret)


def cal_auprc(label_list, score_list):
    precision, recall, thresholds = metrics.precision_recall_curve(np.array(label_list), np.array(score_list))
    auprc = metrics.auc(recall, precision)

    ret = auprc

    return (ret)


def baseline_auprc(label_list, score_list, baseline):
    pos_cnt = label_list.count(1)  # 正样本个数
    neg_cnt = len(label_list) - pos_cnt  # 负样本个数

    label_score_matrix = pd.DataFrame({0: label_list, 1: score_list})

    # 正样本
    pos_label_score_matrix = label_score_matrix.iloc[list(label_score_matrix.iloc[:, 0] == 1), :]

    # 负样本
    neg_label_score_matrix = label_score_matrix.iloc[list(label_score_matrix.iloc[:, 0] != 1), :]

    auprc = -1

    # for current_index in range(0,len(label_list)):
    # if label_list[current_index]== 1:

    # pos_label_list=

    # neg_index=

    if pos_cnt / len(label_list) < baseline:
        neg_choose = int(pos_cnt * 9)
        itera = 100
        auprclist = []
        for i in range(itera):
            pos_choose_matrix = pos_label_score_matrix
            neg_choose_matrix = neg_label_score_matrix.sample(n=neg_choose)  # 随机抽取

            choose_label = pos_choose_matrix.iloc[:, 0].tolist() + neg_choose_matrix.iloc[:, 0].tolist()
            choose_score = pos_choose_matrix.iloc[:, 1].tolist() + neg_choose_matrix.iloc[:, 1].tolist()

            auprc_temp = cal_auprc(choose_label, choose_score)
            auprclist.append(auprc_temp)

        auprc = np.mean(auprclist)

    if pos_cnt / len(label_list) == baseline:
        pos_choose_matrix = pos_label_score_matrix
        neg_choose_matrix = neg_label_score_matrix

        choose_label = pos_choose_matrix.iloc[:, 0].tolist() + neg_choose_matrix.iloc[:, 0].tolist()
        choose_score = pos_choose_matrix.iloc[:, 1].tolist() + neg_choose_matrix.iloc[:, 1].tolist()

        auprc_temp = cal_auprc(choose_label, choose_score)
        auprc = auprc_temp

    if pos_cnt / len(label_list) > baseline:
        pos_choose = int(neg_cnt / 9)
        itera = 100
        auprclist = []
        for i in range(itera):
            pos_choose_matrix = pos_label_score_matrix.sample(n=pos_choose)  # 随机抽取
            neg_choose_matrix = neg_label_score_matrix

            choose_label = pos_choose_matrix.iloc[:, 0].tolist() + neg_choose_matrix.iloc[:, 0].tolist()
            choose_score = pos_choose_matrix.iloc[:, 1].tolist() + neg_choose_matrix.iloc[:, 1].tolist()

            auprc_temp = cal_auprc(choose_label, choose_score)
            auprclist.append(auprc_temp)

        auprc = np.mean(auprclist)

    return (auprc)

