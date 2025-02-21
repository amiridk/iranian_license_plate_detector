from django.urls import path
from .views import home, delete_image

urlpatterns = [
    path('', home, name='home'),
    path('delete/<int:image_id>/', delete_image, name='delete_image'),
]