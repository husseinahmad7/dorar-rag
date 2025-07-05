from django.urls import path
from . import views

app_name = 'hadith'

urlpatterns = [
    path('', views.index, name='index'),
    path('search/', views.search_hadiths, name='search'),
    path('api/hadiths/', views.api_hadiths, name='api_hadiths'),
    path('api/hadiths/<int:hadith_id>/', views.api_hadith_detail, name='api_hadith_detail'),
    path('api/semantic-search/', views.api_semantic_search, name='api_semantic_search'),
    path('api/rag-search/', views.api_rag_search, name='api_rag_search'),
    path('api/hadiths/<int:hadith_id>/recommendations/', views.api_recommended_hadiths, name='api_recommended_hadiths'),
]
