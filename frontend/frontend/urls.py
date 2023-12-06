from django.urls import path
from frontend.views import frontend, search

app_name = 'frontend'

urlpatterns = [
    path('', frontend, name='frontend'),
    path('search', search, name='search'),
]