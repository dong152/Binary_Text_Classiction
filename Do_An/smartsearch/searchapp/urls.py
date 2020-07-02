from django.urls import path
from . import views
from .views import *
urlpatterns = [
    path('', views.index, name='home-page'),
    path('search', SearchView.as_view(), name="google-search-view")
]