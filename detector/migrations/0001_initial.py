from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="DetectionLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("objeto_tipo", models.CharField(max_length=20)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                ("count_total", models.IntegerField()),
            ],
            options={"ordering": ["-timestamp"]},
        ),
    ]
