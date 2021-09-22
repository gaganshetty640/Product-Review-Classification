from django.shortcuts import render
import warnings
warnings.filterwarnings('ignore')

from collaborativebased_api.collab_code import load_data,get_clean_data,get_pivot_table,svd_model,collab_based_model,get_recommended_reviews
from django.http import JsonResponse

def collab_model_training(request):
    collab_df = load_data()
    clean_data = get_clean_data(collab_df)
    users_items_pivot_sparse_matrix,users_items_pivot_matrix_df,users_ids = get_pivot_table(clean_data)
    cf_preds_df= svd_model(users_items_pivot_sparse_matrix,users_items_pivot_matrix_df,users_ids)
    collab_based_model(cf_preds_df)
      
    #print(similar_review_data.shape)
    
    return JsonResponse({'model_training':'completed'})
    #
    
    
def collab_model_testing(request):
    
    try:

        USER_ID=int(request.GET.get('USER_ID'))
        
        print(USER_ID)
        
        Recommended_reviews = get_recommended_reviews(USER_ID)
        Recommended_reviews = list(Recommended_reviews)
        #Recommended_reviews = list(Recommended_reviews['reviews.text'])
        print(type(Recommended_reviews))
        #print(type(Recommended_reviews.to_json()))
        
        return JsonResponse(Recommended_reviews,safe= False)
           
    except KeyError:
        return JsonResponse({USER_ID:"This USER ID has not rated more than 5 products"})
       