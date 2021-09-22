import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
import pickle

def load_data():
    df= pd.read_csv(r"D:/Desktop/Content Personalization/customer_segmentation/contentbased_api/Amazon_sample.csv")
   
    print("data read done.....")
    return df


def get_featured_data(df):
    features = ['brand','categories','manufacturer','reviews.text']
    
    def combine_features(row):
        return row['brand'] + " " + row['categories'] + " " + row['manufacturer'] + " " + row['reviews.text']

    for feature in features:
        df[feature] = df[feature].fillna('')
    
    ##Create a column in DF which combines all selected features

    df['Desc'] = df.apply(combine_features, axis =1)
    featured_data = df.copy()
    print(featured_data.columns)
    return featured_data


def get_clean_data(featured_data):
    
    stop = stopwords.words('english')
    
    df = featured_data.copy()
    df['documents_cleaned'] = df.Desc.str.replace("[^\w\s]", "")
    df['documents_cleaned'] = df.documents_cleaned.apply(lambda x: x.lower())
    df['documents_cleaned'] = df['documents_cleaned'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    df['documents_cleaned'] = df['documents_cleaned'].apply(lambda x: ' '.join( [item for item in x.split() if len(item)>4]))
    df['documents_cleaned'] = df['documents_cleaned'].str.replace('\d+', '')
    df['documents_cleaned'] = df['documents_cleaned'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
    df = df.replace('_____',' ', regex=True)
    
    documents_df=df[["brand","categories","manufacturer","reviews.text","Desc","documents_cleaned"]]
    # documents_df.rename(columns = {'Desc':'documents'}, inplace = True)
    print('get_clean_data completed')
    return documents_df

def model_building(documents_df):
    
    print(documents_df.columns)
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    print('sbert_model')
    
    document_embeddings = sbert_model.encode(documents_df['documents_cleaned'])
    print('document_embeddings')
    
    pairwise_similarities=cosine_similarity(document_embeddings)
    print('pairwise_similarities')
 
    return pairwise_similarities

def content_based_model(pairwise_similarities):
    
    #content_based_model = pairwise_similarities.copy()
    
    f = open('pairwise_similarities.pkl' ,"wb")
    pickle.dump(pairwise_similarities,f)
    
    f.close()
    
    print('pickle file generated')
    
def get_similar_reviews(Review_ID):
    
    pickle_off = open ("pairwise_similarities.pkl", "rb")
    pairwise_similarities = pickle.load(pickle_off)
    
    #print('pickle file loaded')
    sim_df = pd.DataFrame(pairwise_similarities)
    
    #print(sim_df)
    
    most_similar_list = sim_df.iloc[Review_ID].nlargest(10).index.tolist()
    
    #print('top 10 correlated reviews selected')
    
    #reviews_df = pd.read_csv('reviews_df.csv')
    
    #print('read reviews dataframe')
    #Similar_reviews = pd.DataFrame(reviews_df.iloc[most_similar_list]['reviews.text'].reset_index(drop = True))
    #print(Similar_reviews)
    #print('got final dataframe')
    
    return most_similar_list