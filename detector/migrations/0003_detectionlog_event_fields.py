from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("detector", "0002_detectionlog_images"),
    ]

    operations = [
        migrations.AddField(
            model_name="detectionlog",
            name="evento_tipo",
            field=models.CharField(default="counter", max_length=20),
        ),
        migrations.AddField(
            model_name="detectionlog",
            name="track_id",
            field=models.IntegerField(blank=True, null=True),
        ),
    ]

