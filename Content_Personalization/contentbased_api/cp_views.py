from django.shortcuts import render
import warnings
warnings.filterwarnings('ignore')

from contentbased_api.cp_code import load_data,get_featured_data,get_clean_data,model_building,content_based_model,get_similar_reviews
from django.http import JsonResponse

def contentbased_model_training(request):
    Reviewdata = load_data()
    feature_data = get_featured_data(Reviewdata)
    clean_data = get_clean_data(feature_data)
    model_embedding= model_building(clean_data)
    content_based_model(model_embedding)
      
    #print(similar_review_data.shape)
    
    return JsonResponse({'model_training':'completed'})
    #
    
    
def contentbased_model_testing(request):
    
     try:

        Review_ID=int(request.GET.get('Review_ID'))
        
        print(Review_ID)
        
        sim_reviews = get_similar_reviews(Review_ID)
        sim_reviews = list(sim_reviews)
        #sim_reviews = list(sim_reviews['reviews.text'])
        print(type(sim_reviews))
        #print(type(sim_reviews.to_json()))
        
        return JsonResponse(sim_reviews,safe= False)
    
     except IndexError:
        return JsonResponse({Review_ID:"Invalid Review_ID"})

       