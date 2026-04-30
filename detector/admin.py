from django.contrib import admin

from detector.models import DetectionLog


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ("id", "objeto_tipo", "timestamp", "count_total")
    list_filter = ("objeto_tipo", "timestamp")
