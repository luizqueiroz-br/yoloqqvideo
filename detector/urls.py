from django.urls import path
from django.contrib.auth import views as auth_views

from detector import views

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", auth_views.LoginView.as_view(template_name="detector/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("registros/", views.detection_logs, name="detection_logs"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("api/frame/", views.frame_api, name="frame_api"),
    path("api/counters/", views.counters_api, name="counters_api"),
    path("api/model-config/", views.model_config_api, name="model_config_api"),
    path("api/hf-models/", views.hf_models_api, name="hf_models_api"),
    path("api/set-model-config/", views.set_model_config_api, name="set_model_config_api"),
    path("api/line-config/", views.line_config_api, name="line_config_api"),
    path("api/sources/", views.list_sources_api, name="list_sources_api"),
    path("api/select-source/", views.select_source_api, name="select_source_api"),
    path("api/set-line-config/", views.set_line_config_api, name="set_line_config_api"),
]
