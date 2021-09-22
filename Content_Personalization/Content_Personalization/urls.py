"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from contentbased_api.cp_views import contentbased_model_training,contentbased_model_testing
from collaborativebased_api.collab_views import collab_model_training,collab_model_testing

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('content_model_train/',contentbased_model_training),
    path('get_similar_reviews/',contentbased_model_testing),
    path('collab_model_train/',collab_model_training),
    path('get_recommended_reviews/',collab_model_testing)
    
    
]
