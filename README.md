# Paper-Classification

**Graph convolutional network **
for paper classification (7 classes) using Pytorch. 
Finished the code in gnn_main.py and gnn_utils.py

Dataset in the datasets folder:
cora.content – paper information (paper id, paper word vector, paper label)
cora.cites – citation relationships of different papers (id of cited paper, id of citing
paper)

Finished GCN class and GraphConvolution class for constructing GCN model in gnn_utils.py. 

GCN model detail: 2 layers in total. The hidden size = 32. 
Use relu activation in the hidden layer, and set dropout ratio = 0.5.

Finished model_train function for GCN model training (using train data) with Adam optimization in gnn_main.py.

Finished model_test function for model testing (using test data) in gnn_main.py.
