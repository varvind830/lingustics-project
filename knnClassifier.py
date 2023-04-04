from itertools import chain
import pandas as pd
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import string
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import numpy as np
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
nltk.download('wordnet')

# Read the data
def read_data():
    
    data = pd.read_csv("dataframe.csv")
    data["dependencies"].fillna("", inplace=True)
    data["relevant_dependencies"].fillna("", inplace=True)
    data["dependencies"] = data["dependencies"].apply(lambda x: x.split("),"))
    data["relevant_dependencies"] = data["relevant_dependencies"].apply(lambda x: x.split("),"))
    data.drop('recommended_exclusion', axis=1, inplace=True)
    data.drop('sentence_id', axis=1, inplace=True)
    
    return data.iloc[:40000,:], data.iloc[40000:,:]

train, test = read_data()

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

train_raw = process_data(train)
test_raw = process_data(test)
#print(train_raw[:5])

def count_identifiers(data):
    identifier_counts = {}
    for item in data:
        dependencies = item[2:]
        for dependency in dependencies:
            identifier = dependency[0]
            identifier_counts[identifier] = identifier_counts.get(identifier, 0) + 1
    return identifier_counts

dependencies = count_identifiers(train_raw).keys()

#print(dependencies)

def makeFeatures(item):
    
    features = {f: False for f in dependencies}
    
    word = item[1]
    
    for feature in item[2:]:
        features[feature[0]] = True
    
    return features


X_train = [makeFeatures(s) for s in train_raw]
y_train = [s[0] for s in train_raw]

x_train_array = np.array([[sample[feature_name] for feature_name in feature_names] for sample in X_train])

X_test = [makeFeatures(s) for s in test_raw]
y_test = [s[0] for s in test_raw]

x_test_array = np.array([[sample[feature_name] for feature_name in feature_names] for sample in X_test])


knn = KNeighborsClassifier(n_neighbors=5, metric='jaccard')

feature_names = list(count_identifiers(train_raw).keys())

# Convert the list of dictionaries to a numpy array
x_train_array = np.array([[sample[feature_name] for feature_name in feature_names] for sample in X_train])
print(x_train_array[:2])


#we may need to fit the data idk

knn.fit(x_train_array, y_train)


y_pred = knn.predict(x_test_array)

#tofix:
#metrics.flat_f1_score(y_test, y_pred, average='weighted')

#do analysis