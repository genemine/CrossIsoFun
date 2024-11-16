# CrossIsoFun
## 1. Description
CrossIsoFun is a multi-omics data integration framework for isoform function prediction. It utilizes an autoencoder trained within the cycleGAN framework to generate IIIs from expression profiles, sequence features, and PPIs. Then, GCNs are employed for omics-specific learning, utilizing omics features and sample correlations to derive initial predictions from expression, sequence, and III data, respectively. Finally, it applies VCDN on the predicted label distributions obtained from GCNs to explore the cross-omics correlations. This strategy facilitates effective multi-omics integration and final function prediction. 


## 2. Input data
The expression profiles, sequence features, and PPI data of isoforms are required as input for CrossIsoFun. The demo input data are provided in the folder 'data_demo', which includes training data for building models and test data for evaluating the performance of CrossIsoFun.


## 3. Implementation
CrossIsoFun is implemented in Python. It is tested on both MacOS and Linux operating systems. They are freely available for non-commercial use.


## 4.Dependency

Python >= 3.7.11

torch >= 2.0.1

numpy >= 1.24.3

pandas >= 3.8.5

scikit-learn >= 1.3.0

torch_scatter >= 2.1.2+pt20cu117

## 5. Usage
We provide a demo script to show how to run CrossIsoFun. Run the following command from the command line:

```bash
python CrossIsoFun.py ./data_demo/train_feature/ ./data_demo/test_feature/ ./data_demo/train_label_folder/ ./data_demo/output/
```

With this command, you can straightforwardly implement and apply CrossIsoFun. It will first train a model and then make predictions on the demo data. Specifically, the input for the script includes four directories:

['./data_demo/train_feature/'](./data_demo/train_feature/) contains feature files for the training dataset, including `iso_expr.txt`, `iso_seqdm.txt`, `iso_gene.txt`, and `train_isoform_list.txt`. Specifically:
- `iso_expr.txt` contains the expression profiles of isoforms in the training set.
- `iso_seqdm.txt` contains the sequence features of isoforms in the training set.
- `iso_gene.txt` records the mapping relationships between isoforms and genes in the training set.
- `train_isoform_list.txt` lists the isoforms used as training samples

['./data_demo/train_labels/']('./data_demo/train_labels/') contains files providin information about the GO annotations used as training labels, including `num_GO_map.txt`, and the `goterms/` directory. Specifically:
- `num_GO_map.txt` records the mapping from GO terms to their indices in the experiment
- The `goterms/` directory contains the files listing the genes annotated to the GO terms. e.g. `GO_0000278.txt` provides a list genes annotated to GO:0000278. 

['./data_demo/test_feature/'](./data_demo/train_feature/) contains feature files for the testing dataset, including `iso_expr.txt`, `iso_seqdm.txt`, `iso_gene.txt`, and `train_isoform_list.txt`. Specifically:
- `iso_expr.txt` contains the expression profiles of isoforms in the testinging set.
- `iso_seqdm.txt` contains the sequence features of isoforms in the testinging set.
- `iso_gene.txt` records the mapping relationships between isoforms and genes in the testinging set.
- `train_isoform_list.txt` lists the isoforms used as testing samples

['./data_demo/output/'](./data_demo/output/) is the directory used to store the output file `iso_score.txt`. Specifically, `iso_score.txt` is a matrix containing the prediction scores obtained from CrossIsoFun. Each row of the matrix corresponds to an isoform and each column corresponds to a GO term (function). e.g. X<sub>ij</sub> denotes the predicted probability that isoform i is annotated to the GO term j.

## 6. Contact
If any questions, please do not hesitate to contact me at:
<br>
Yiwei Liu, `ywlmicky@csu.edu.cn`
<br>
Jianxin Wang, `jxwang@mail.csu.edu.cn`
