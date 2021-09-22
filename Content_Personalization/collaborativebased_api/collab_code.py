import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pickle


def load_data():
    collab_df = pd.read_csv(r"D:/Desktop/Content_Personalization/collaborativebased_api/Amazon_sample.csv")
   
    print("data read done.....")
    return collab_df

def get_clean_data(collab_df):
    
    collab_df = collab_df.dropna(subset=['reviews.id','reviews.username'])
    
    collab_df["User_ID"]= pd.Categorical(collab_df['reviews.username'])
    collab_df["User_ID"] = collab_df["User_ID"].cat.codes

    collab_df = collab_df.rename(columns={'reviews.id': 'Content_ID','reviews.rating':'rating'})

    x = collab_df['User_ID'].value_counts().to_dict()

    collab_df['rating_per_user'] = collab_df['User_ID'].replace(x)
    
    rating =collab_df[collab_df['rating_per_user']>=5]

    rating = rating.drop_duplicates(subset=['Content_ID', 'User_ID'], keep='first')
    
    rating =collab_df[collab_df['rating_per_user']>=5]
   
    rating.to_csv('rating.csv')
    
    print('get_clean_data done')
    return rating

def get_pivot_table(rating):
    
    #clean_data = pd.read_csv('clean_data.csv')
    
    users_items_pivot_matrix_df = rating.pivot_table(index='User_ID', 
                                                        columns='Content_ID', 
                                                        values='rating').fillna(0)
    
    users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
    
    users_ids = list(users_items_pivot_matrix_df.index)

    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
    
    print('get_pivot_table done')

    return users_items_pivot_sparse_matrix,users_items_pivot_matrix_df,users_ids

def svd_model(users_items_pivot_sparse_matrix,users_items_pivot_matrix_df,users_ids):
    
    NUMBER_OF_FACTORS_MF = 10
    
    #Performs matrix factorization of the original user item matrix
    #U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
    
    U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
    print("U", U.shape)
    print("Vt", Vt.shape)
    sigma = np.diag(sigma)
    print("sigma", sigma.shape)


    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
    #print(all_user_predicted_ratings.shape)
       
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())


    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
    #all_user_predicted_ratings_norm
    
    
    #Converting the reconstructed matrix back to a Pandas dataframe
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
    #cf_preds_df.head(5)
    
    print('svd_model done')
    
    return cf_preds_df

def collab_based_model(cf_preds_df):
    
    f = open('cf_preds_df.pkl' ,"wb")
    pickle.dump(cf_preds_df,f)
    f.close()
    
    print('pickle file generated')

def get_recommended_reviews(USER_ID):
    
    pickle_off = open ("cf_preds_df.pkl", "rb")
    cf_preds_df = pickle.load(pickle_off)
    
    rating = pd.read_csv('rating.csv')
    

    class CFRecommender:  ## class with namee CFRecommender is created and then its atttributes are defined
        
        MODEL_NAME = 'Collaborative Filtering'
        
        # instance attributes
        def __init__(self, cf_predictions_df, items_df=None): ### this is initializer method
            self.cf_predictions_df = cf_predictions_df
            self.items_df = items_df
            
        # Creating methods. Methods are functions defined inside the body of a class. They are used to define the behaviors of an object.        
        def get_model_name(self):
            return self.MODEL_NAME
         
        def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
            # Get and sort the user's predictions
            sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().\
                                     rename(columns={user_id: 'recStrength'})
    
            # Recommend the highest predicted rating movies that the user hasn't seen yet.
            recommendations_df = sorted_user_predictions[~sorted_user_predictions['Content_ID'].isin(items_to_ignore)].\
                               sort_values('recStrength', ascending = False).head(topn)
    
            if verbose:
                if self.items_df is None:
                    raise Exception('"items_df" is required in verbose mode')
    
                recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                              left_on = 'Content_ID', 
                                                              right_on = 'Content_ID')[['recStrength', 'Content_ID']]
    
    #         recommendations_df = recency_of_recommendation_func(recommendations_df, user_id, 'recStrength')
                    
            return recommendations_df
    
    
    # instantiate the object
    cf_recommender_model = CFRecommender(cf_preds_df)
    
    
    Rec_df = cf_recommender_model.recommend_items(USER_ID, items_to_ignore=list(rating[rating["User_ID"]==USER_ID]['Content_ID'].unique()), topn=10, verbose=False)
    
    ### content_id list
    most_similar_list = Rec_df.nlargest(10,'recStrength').Content_ID.tolist()
    most_similar_list = [int(x) for x in most_similar_list]
    #print('top 10 correlated reviews selected')
    
    rating = pd.read_csv('rating.csv')
    print('read rating dataframe')
    
    
    ###Content id df
    Recommended_reviews = pd.DataFrame(rating[rating["Content_ID"].isin(most_similar_list)][['reviews.text']].reset_index(drop= True))
    print(Recommended_reviews)
    
    print('got Recommended_reviews')
    
    return most_similar_list
    #return Recommended_reviews