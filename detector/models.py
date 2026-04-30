from django.db import models


class DetectionLog(models.Model):
    objeto_tipo = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)
    count_total = models.IntegerField()
    evento_tipo = models.CharField(max_length=20, default="counter")
    track_id = models.IntegerField(null=True, blank=True)
    imagem_full = models.ImageField(upload_to="detections/full/", null=True, blank=True)
    imagem_crop = models.ImageField(upload_to="detections/crop/", null=True, blank=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.evento_tipo}:{self.objeto_tipo} - {self.count_total} - {self.timestamp}"
