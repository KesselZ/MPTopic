import umap
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.cluster import Birch
import seaborn as sns
from collections import Counter
import hdbscan
from sklearn_extra.cluster import KMedoids
from sentence_transformers import SentenceTransformer

def di_reduction(vec,dimensions):
    # pca = PCA(n_components=dimensions)
    # d_result= pca.fit_transform(vec)

    # ts = TSNE(n_components=dimensions, perplexity=45)
    # d_result = ts.fit_transform(vec)

    reducer = umap.UMAP(random_state=42,n_neighbors=15,n_components=dimensions,min_dist=0.1)
    d_result = reducer.fit_transform(vec)


    return d_result

def di_reduction2(vec,dimensions):
    # pca = PCA(n_components=dimensions)
    # d_result= pca.fit_transform(vec)

    reducer = umap.UMAP(n_neighbors=15,n_components=dimensions,metric='cosine')
    d_result = reducer.fit_transform(vec)

    # ts = TSNE(n_components=dimensions, perplexity=45)
    # d_result = ts.fit_transform(vec)
    return d_result

def transfer_to_feature(example,model_name='sentence-transformers/all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(example,show_progress_bar=True)
    return embeddings


def train(classes,d_result,choose,db_para=55):
    if(choose=='kmedoid'):
        clustering_model = KMedoids(n_clusters=classes)
        clustering_model.fit(d_result)
    elif(choose=='birch'):
        clustering_model = Birch(n_clusters=classes)
        clustering_model.fit(d_result)
    elif (choose == 'kmeans'):
        clustering_model = KMeans(n_clusters=classes)
        clustering_model.fit(d_result)
    elif (choose == 'hdbscan'):
        clustering_model = hdbscan.HDBSCAN(min_cluster_size=db_para,metric='euclidean')
        clustering_model.fit(d_result)
    else:
        raise Exception("You choosed a wrong clustering algorithm, please check 'clustering_choose' parameter")

    cluster_assignment = clustering_model.labels_
    return cluster_assignment


def evaluate2(cluster_assignment,target,classes):
    evaluation = []
    count=0
    for i in range(classes):
        evaluation.append([])

    for i in range(classes):
        for r in range(cluster_assignment.size):
            if(i==cluster_assignment[r]):
                evaluation[i].\
                    append(target[r])

    for i in range(classes):
        if(len(evaluation[i])!=0):count=count+1


    answer = Counter(target)
    print("Correct answer(target): ",answer)
    list=[]
    for i in range(classes):
        result2 = Counter(evaluation[i])
        list.append(result2)
        print("cluster",i,": ",result2)
    purity_value,outlier_num=analysis(answer,list)
    return purity_value,count,outlier_num

def analysis(answer,result2):
    sum1=0
    sum2=0
    sum3=0

    for i in answer:
        sum1= sum1 + answer[i]

    for i in range(len(result2)):
        if(len(list(result2[i]))):
            sum2=sum2+max(result2[i].values())

    for i in range(len(result2)):
        if(len(list(result2[i]))):
            sum3=sum3+sum(result2[i].values())
    print("The number of document that is not outliers:", sum3)
    print("The number of document:", sum1)
    print("The number of guessed correct document:",sum2)
    print(sum2 / sum3)
    purity_value=sum2 / sum3
    outlier_num=sum1-sum3
    return purity_value,outlier_num

def plotPic(classes,d_result,cluster_assignment):
    X = []
    Y = []
    for i in range(classes):
        X.append([])
        Y.append([])

    for i in range(len(d_result)):
        X[cluster_assignment[i]].append(d_result[i][0])
        Y[cluster_assignment[i]].append(d_result[i][1])

    for i in range(classes):
        plt.scatter(X[i], Y[i],s=50)

    plt.title("The visulization")
    plt.show()


def plotPic2(classes,d_result,cluster_assignment):
    X = []
    Y = []


    for i in range(len(d_result)):
        X.append(d_result[i][0])
        Y.append(d_result[i][1])

    scatter=plt.scatter(X, Y,c=cluster_assignment, s=0.05, cmap='Spectral')
    # plt.legend(handles=scatter.legend_elements()[0])
    plt.title("The scatter graph of document clustering")
    plt.show()

def concat(text,cluster_assignment,classes):
    print(cluster_assignment)
    long_text = []
    for i in range(classes):
        long_text.append('')

    for r in range(len(cluster_assignment)):
        long_text[int(cluster_assignment[r])]=long_text[int(cluster_assignment[r])]+text[r]
    print(type(long_text))
    return long_text
