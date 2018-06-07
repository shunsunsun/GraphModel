# PGRR

Mobile Access Record Resolution on Large-Scale Identifier-Linkage Graphs (KDD-18)

This is the Spark implementations of two versions (unsupervised and semi-supervised) of Parallel Graph-based Record Resolution (PGRR) algorithm, which solves the Mobile Access Records Resolution (MARR) problem.


## Requirement

* Spark
* Scala
* ODPS
* Maven

## Basic Usage

### Input Data
For unsupervised version, each dataset contains 2 table: features and edges.
```
1. features: each record contains n+1 fields.
node_1 feature_1 feature_2 ... feature_n
node_2 feature_1 feature_2 ... feature_n
...

2. edges: each record contains two connected nodes.
node_1 node_2
node_2 node_3
...
```
For semi-supervised version, each dataset contains 3 table: features, edges and labels.
```
1. features: each record contains n+1 fields.
node_1 feature_1 feature_2 ... feature_n
node_2 feature_1 feature_2 ... feature_n
...

2. edges: each line contains two connected nodes.
node_1 node_2
node_2 node_3
...

3. labels: each record contains a node and its label.
node_1 label_1
node_2 label_2
...
```

### Run
To run PGRR, recommend using `IntelliJ IDEA` to open this project.

