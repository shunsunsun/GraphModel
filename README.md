# Graph Model
Large scale graphical modeling has been playing a key role in assuring success in many of the challenges in big data companies, as many problems can be abstracted and solved by graphical models. Recently, methods based on representing networks in the vector space, while preserving their properties, have become widely popular. The embeddings are inputs as features to a model and the parameters are learned based on the training data. Obtaining a vector representation of a large heterogeneous graph is inherently difficult and poses several challenges where related research topics may arise:   

1. Heterogeneity: A “good” representation of nodes should preserve not only the structure of the graph but also the entity differences. Existing works mainly focus on embeddings of homogeneous graphs but majority of problems should be solved with heterogeneous graphical modeling. 

2. Scalability: The graph formed by buyers, sellers and commodities contains hundreds of billions of nodes and edges. Defining a scalable model can be challenging especially when the model is aimed to preserve global properties of the graph.  

3. Reliability: Observed edges or node labels can be polluted and thus are not reliable.   

To solve the above mentioned challenges, we propose a principled effort to investigate a new framework of deep graph that seamlessly integrates computing, learning and inference in a consistent and optimum set up. Current checked in methods and papers include the following list which is still growing:  

1. ANRL: Attributed Network Representation Learning via Deep Neural Networks (IJCAI-18)

2. DELP: Semi-Supervised Learning Through Dynamic Graph Embedding and Label Propagation (Under Review) 

3. PGRR: Mobile Access Record Resolution on Large-Scale Identifier-Linkage Graphs (KDD-18)

4. PRRE: Personalized Relation Ranking Embedding for A ributed Network (Under Review) 


