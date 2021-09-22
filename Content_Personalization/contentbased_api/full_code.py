import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
import pickle


df= pd.read_csv(r"D:/Desktop/Content Personalization/customer_segmentation/cp_api/Amazon_sample.csv")

print("data read done.....")

features = ['brand','categories','manufacturer','reviews.text']

def combine_features(row):
    return row['brand'] + " " + row['categories'] + " " + row['manufacturer'] + " " + row['reviews.text']

for feature in features:
    df[feature] = df[feature].fillna('')

##Create a column in DF which combines all selected features

df['Desc'] = df.apply(combine_features, axis =1)

from nltk.corpus import stopwords
stop = stopwords.words('english')

df['documents_cleaned'] = df.Desc.str.replace("[^\w\s]", "")
df['documents_cleaned'] = df.documents_cleaned.apply(lambda x: x.lower())
df['documents_cleaned'] = df['documents_cleaned'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
df['documents_cleaned'] = df['documents_cleaned'].apply(lambda x: ' '.join( [item for item in x.split() if len(item)>4]))
df['documents_cleaned'] = df['documents_cleaned'].str.replace('\d+', '')
df['documents_cleaned'] = df['documents_cleaned'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
df = df.replace('_____',' ', regex=True)


documents_df=df[["brand","categories","manufacturer","reviews.text","Desc","documents_cleaned"]] 

from sentence_transformers import SentenceTransformer   

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

document_embeddings = sbert_model.encode(documents_df['documents_cleaned'])