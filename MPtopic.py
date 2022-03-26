from sklearn.datasets import fetch_20newsgroups
import processor
from sklearn import metrics
import time_count
import tfidf

# data is the corpus in form of strings in a list
# target is the labbled data result
# classes is the target clusters if your algorithm is not hdbscan
# db_para is the min_cluster_size if you use hdbscan
# dimensions is the target dimension reduction before clustering
def MPtopic(data,target,classes,db_para=35,dimensions=-1,choose_clustering='kmeans'):
    #get features
    time_count.time_report()
    print("getting features....")
    vec=processor.transfer_to_feature(data)
    time_count.time_report()

    #get first reduction
    print("doing first reduction....")
    if(dimensions != -1):vec=processor.di_reduction2(vec,dimensions)
    time_count.time_report()

    #clustering
    print("clustering....")
    cluster_assignment=processor.train(classes,vec,choose_clustering,db_para)
    tfidf.topic_modelling(data,cluster_assignment)
    time_count.time_report()

    #dimension reduction for visulization
    print("calculating 2d picture...")
    result_2d=processor.di_reduction(vec,2)
    time_count.time_report()

    #Evaluation and visulization
    print("evaluating...")
    # processor.evaluate(cluster_assignment,target)
    processor.evaluate2(cluster_assignment,target,classes)
    processor.plotPic2(classes,result_2d,cluster_assignment)

    print(cluster_assignment)
    print("Silhouette Coefficient：", metrics.silhouette_score(vec, cluster_assignment, metric='euclidean'))
    time_count.time_report()

