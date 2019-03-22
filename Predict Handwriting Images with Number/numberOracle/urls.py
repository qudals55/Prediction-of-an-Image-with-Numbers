from django.conf.urls import url
from . import views

urlpatterns = [
    # ex: /numberOracle/
    url(r'^$', views.index, name='index'),
    # ex: /numberOracle/ajaxform/
    url(r'^ajaxform/$', views.ajaxform, name='ajaxform'),
    # ex: /numberOracle/searchData/
    url(r'^searchData/$', views.searchData, name='searchData'),
]
