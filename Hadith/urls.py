from django.urls import path
from . import views

app_name = 'hadith'

urlpatterns = [
    path('', views.index, name='index'),
    path('search/', views.search_hadiths, name='search'),
    path('agentic-stream-demo/', views.agentic_stream_demo, name='agentic_stream_demo'),
    path('debug-agentic/', views.debug_agentic_service, name='debug_agentic_service'),
    path('api/hadiths/', views.api_hadiths, name='api_hadiths'),
    path('api/hadiths/<int:hadith_id>/', views.api_hadith_detail, name='api_hadith_detail'),
    path('api/semantic-search/', views.api_semantic_search, name='api_semantic_search'),
    path('api/rag-search/', views.api_rag_search, name='api_rag_search'),
    path('api/agentic-search/', views.api_agentic_search, name='api_agentic_search'),
    path('api/agentic-search-stream/', views.api_agentic_search_stream, name='api_agentic_search_stream'),
    path('api/hadiths/<int:hadith_id>/recommendations/', views.api_recommended_hadiths, name='api_recommended_hadiths'),
]
