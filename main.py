import MPtopic
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(subset='all')
data = dataset['data']
target= dataset['target']

MPtopic.MPtopic(data=data,target=target,classes=20)