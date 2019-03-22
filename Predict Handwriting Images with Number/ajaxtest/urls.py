from django.conf.urls import url 
from . import views
 
urlpatterns = [
    # ex: /ajaxtest/
    url(r'^$',views.index, name = 'index'),
    # ex: /ajaxtest/ajaxform/
    url(r'^ajaxform/$', views.ajaxform, name='ajaxform'),
    # ex: /ajaxtest/searchData/
    url(r'^searchData/$', views.searchData, name='searchData'),
]
