#pip install corextopic
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
import pandas as pd

#https://ryanjgallagher.github.io/code/corex/example  <- main documantation
#https://github.com/gregversteeg/CorEx
#https://stackoverflow.com/questions/48521740/using-mca-package-in-python/57237247#57237247
#https://stats.stackexchange.com/questions/159705/would-pca-work-for-boolean-binary-data-types
def read_data():
    
    data = pd.read_csv("dataframe.csv")
    data["dependencies"].fillna("", inplace=True)
    data["relevant_dependencies"].fillna("", inplace=True)
    data["dependencies"] = data["dependencies"].apply(lambda x: x.split("),"))
    data["relevant_dependencies"] = data["relevant_dependencies"].apply(lambda x: x.split("),"))
    data.drop('recommended_exclusion', axis=1, inplace=True)
    data.drop('sentence_id', axis=1, inplace=True)
    
    return data

data = read_data()

def process_data(data):
    dataset = []
    feature_set = []
    for i, (position, word, gerund, tags, dependencies, relDependencies, sentence) in data.iterrows():
        feature_set.append(gerund)
        feature_set.append(word)    
        for dependency in relDependencies:
            if len(dependency.split())<3:
                continue
            id = dependency.split()[0]
            w1 = dependency.split()[1][1:-1].split("-")[0]
            p1 = dependency.split()[1][1:-1].split("-")[1]
            w2 = dependency.split()[2].split("-")[0]
            p2 = dependency.split()[2].split("-")[1]
            organized_dependency = (id, w1, p1, w2, p2)
            feature_set.append(organized_dependency)  
        dataset.append(feature_set)
        feature_set = []
    return dataset


def count_identifiers(data):
    identifier_counts = {}
    for item in data:
        dependencies = item[2:]
        for dependency in dependencies:
            identifier = dependency[0]
            identifier_counts[identifier] = identifier_counts.get(identifier, 0) + 1
    return identifier_counts

def makeArray(item):
    features = {f: 0 for f in dependencies}
    for feature in item[2:]:
        features[feature[0]] = 1
    return features

goodData = process_data(data)
#print(goodData[:3])

dependencies = count_identifiers(goodData).keys()
#print(dependencies)
X = [makeArray(s) for s in goodData]
tags = [s[0] for s in goodData]

xArray = np.array([[int(sample[feature_name]) for feature_name in dependencies] for sample in X], dtype=int)
#print(xArray[:3])
#print(tags[:3])

# Define a matrix where rows are samples (docs) and columns are features (words)
# Sparse matrices are also supported
xArray = ss.csr_matrix(X)


# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=2)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=dependencies, docs=tags)
