# CrossIsoFun
## 1. Description
CrossIsoFun is a multi-omics data integration framework for isoform function prediction. It utilizes an autoencoder trained within the cycleGAN framework to generate IIIs from expression profiles, sequence features, and PPIs. Then, GCNs are employed for omics-specific learning, utilizing omics features and sample correlations to derive initial predictions from expression, sequence, and III data, respectively. Finally, it applies VCDN on the predicted label distributions obtained from GCNs to explore the cross-omics correlations. This strategy facilitates effective multi-omics integration and final function prediction. 


## 2. Input data
The expression profiles, sequence features, and PPI data of isoforms are required as input for CrossIsoFun. The demo input data are provided in the folder 'data_demo', which includes training data for building models and test data for evaluating the performance of CrossIsoFun.


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
python my_main.py
```

With this command, we will first build a model on the training data and then make predictions on the test data.


## 6. Contact
If any questions, please do not hesitate to contact me at:
<br>
Yiwei Liu, `ywlmicky@csu.edu.cn`
<br>
Jianxin Wang, `jxwang@mail.csu.edu.cn`
