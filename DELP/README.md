# DELP

This is a Tensorflow implementation of the DELP algorithm, which jointly performs graph embedding and label propagation.

## Requirements
* tensorflow
* networkx
* numpy
* scipy
* scikit-learn

All required packages are defined in requirements.txt. To install all requirement, just use the following commands:
```
pip install -r requirements.txt
```

## Basic Usage

### Input Data 
Each dataset contains 3 files: edgelist, features and labels.
```
1. citeseer.edgelist: each line contains two connected nodes.
node_1 node_2
node_2 node_3
...

2. citeseer.feature: this file has n+1 lines.
The first line has the following format:
node_number feature_dimension
The next n lines are as follows: (each node per line ordered by node id)
(for node_1) feature_1 feature_2 ... feature_n
(for node_2) feature_1 feature_2 ... feature_n
...

3. citeseer.label: each line represents a node and its class label.
node_1 label_1
node_2 label_2
...
```

### Run
To run DELP, just execute the following command for node classification task:
```
python main.py
```