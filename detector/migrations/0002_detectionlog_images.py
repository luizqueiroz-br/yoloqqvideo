from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("detector", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="detectionlog",
            name="imagem_crop",
            field=models.ImageField(blank=True, null=True, upload_to="detections/crop/"),
        ),
        migrations.AddField(
            model_name="detectionlog",
            name="imagem_full",
            field=models.ImageField(blank=True, null=True, upload_to="detections/full/"),
        ),
    ]
