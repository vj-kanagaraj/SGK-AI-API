from django.shortcuts import render
from . import views
from django.urls import path, include , re_path
from rest_framework.authtoken.views import obtain_auth_token
from django.views.generic.base import RedirectView
from django.contrib import admin
from django.views.generic import TemplateView
from django.conf.urls import include, url
from django.views.static import serve


urlpatterns = [
    re_path(r"msd\/$", views.msd, name='msd'),
    re_path(r"ferrero\/$", views.ferrero, name='ferrero'),
    re_path(r"excel_extraction\/$", views.excel_extraction, name='excel_extraction'),
    re_path(r"dg\/$", views.dollar_general, name='dollar_general'),
    re_path(r"lang_detect\/$",views.language_detection,name='lang_detect'),
    re_path(r"api_token\/$",obtain_auth_token,name='auth_token'),
    # re_path(r"dataset_update\/$",views.dataset_to_mangodb,name='dataset_update'),
    ]