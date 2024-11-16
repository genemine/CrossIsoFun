# CrossIsoFun
## 1. Description
CrossIsoFun is a multi-omics data integration framework for isoform function prediction. It utilizes an autoencoder trained within the cycleGAN framework to generate IIIs from expression profiles, sequence features, and PPIs. Then, GCNs are employed for omics-specific learning, utilizing omics features and sample correlations to derive initial predictions from expression, sequence, and III data, respectively. Finally, it applies VCDN on the predicted label distributions obtained from GCNs to explore the cross-omics correlations. This strategy facilitates effective multi-omics integration and final function prediction. 


## 2. Input data
The expression profiles, sequence features, and PPI data of isoforms are required as input for CrossIsoFun. The demo input data are provided in the folder 'data_demo', which includes training data for building models and test data for evaluating the performance of CrossIsoFun.
（1) training data
    ./data_demo/train_feature/iso_expr.txt -- expression profiles of isoforms in training dataset
    ./data_demo/train_feature/iso_seqdm.txt -- sequence features of isoforms in training dataset
（2) training labels
    ./data_demo/train_labels/num_GO_map.txt -- mapping from GO terms to their indices in the experiment
    ./data_demo/train_feature/goterms/GO_XXXXXXX.txt -- genes annotated to the GO term GO:XXXXXXX 
（3) testing data
    ./data_demo/test_feature/iso_expr.txt -- expression profiles of isoforms in testing dataset
    ./data_demo/test_feature/iso_seqdm.txt -- sequence features of isoforms in testing dataset
（4) PPI data
    ./data_demo/iso_PPI_demo.txt

## 3. Implementation
CrossIsoFun is implemented in Python. It is tested on both MacOS and Linux operating systems. They are freely available for non-commercial use.


## 4.Dependency

Python > 3.7.11

torch > 2.0.1

numpy > 1.24.3

pandas > 3.8.5

scikit-learn > 1.3.0

torch_scatter > 2.1.2+pt20cu117

## 5. Usage
We provide a demo script to show how to run CrossIsoFun. To test CrossIsoFun on an independent test dataset, run the following command from the command line:

```bash
python CrossIsoFun.py ./data_demo/train_feature/ ./data_demo/test_feature/ ./data_demo/train_label_folder
```

With this command, we will first build a model on the training data and then make predictions on the test data. 
The output will be text file storing a matrix (m*n) denoting the prediction scores of isoforms, where m is the number of isoforms in testing set,
n is the number of predicted functions (GO terms), and the entry (Xij) denotes the predicted probability of the isoform i being annotated to the function j.
The outpput file is named 'iso_score.txt' and will be in the ./data_demo/output.


## 6. Contact
If any questions, please do not hesitate to contact me at:
<br>
Yiwei Liu, `ywlmicky@csu.edu.cn`
<br>
Jianxin Wang, `jxwang@mail.csu.edu.cn`
