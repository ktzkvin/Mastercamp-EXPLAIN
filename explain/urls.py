from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.home, name='home'),
    path('import/', views.import_patent, name='import_patent'),
    path('result/<int:pk>/', views.classification_result, name='classification_result'),
    path('explain/<int:index>/', views.explain_view, name='explain_view'),
]
