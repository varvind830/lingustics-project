import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('dataframe.csv')
df = df[['relevant_dependencies']]
pd.set_option('display.max_colwidth', None)
df['relevant_dependencies'] = df['relevant_dependencies'].fillna('')
df['dependencies'] = df['relevant_dependencies'].str.split(')')
max_deps = max(df['dependencies'].apply(len))
col_names = [f'dep{i+1}' for i in range(max_deps)]
new_df = pd.DataFrame(df['dependencies'].tolist(), columns=col_names)
new_df = new_df.fillna("")
merged_df = pd.concat([df, new_df], axis=1)
merged_df = merged_df.drop(['relevant_dependencies', 'dependencies'], axis=1)

df = merged_df

text_columns = ['dep1', 'dep2', 'dep3', 'dep4', 'dep5', 'dep6', 'dep7', 'dep8', 'dep9', 'dep10', 'dep11', 'dep12', 'dep13', 'dep14', 'dep15', 'dep16', 'dep17', 'dep18', 'dep19']  
texts = df[text_columns].apply(lambda x: ' '.join(x), axis=1)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])
preprocessed_texts = pipeline.fit_transform(texts)

kmeans = KMeans(n_clusters=5)
kmeans.fit(preprocessed_texts)


labels = kmeans.predict(preprocessed_texts)



