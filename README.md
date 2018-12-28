# KMeans-Emails-Clustering-Visualization-NLP
KMeans-Emails-Clustering-Visualization-NLP: KMeans is used to cluster the emails. The words in the contents of emails are tokenlized and stemmed. This project transforms the corpus into vector space using tf-idf.By multidimensional scaling, the clustering result is visualized.

The inputs are a directory (./email) of .eml files.
The outputs are the clusters, the mapping of email-cluster, the key words for each cluster and the 2-D visualized figure.

Details can be found in the source code. This work can be further applied to email filtering and message detection.

Including the following contents, some of which refers to the *[Document Clustering with Python](http://brandonrose.org/clustering)* 

    extract contents from email
    tokenizing and stemming each synopsis
    transforming the corpus into vector space using tf-idf
    calculating cosine distance between each document as a measure of similarity
    clustering the documents using the k-means algorithm
    using multidimensional scaling to reduce dimensionality within the corpus
    plotting the clustering output using matplotlib and mpld3

This Python project relies on the following packages: numpy, pandas, nltk, sklearn, mpld3, email, matplotlib 

![image](https://github.com/zslwyuan/KMeans-Emails-Clustering-Visualization-NLP/blob/master/visual_img_cluster/visualization.png)


![image](https://github.com/zslwyuan/KMeans-Emails-Clustering-Visualization-NLP/blob/master/visual_img_cluster/visualization0.png)
