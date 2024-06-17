# Generated by Django 5.0.6 on 2024-06-17 08:01

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Lipstick",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("brand", models.CharField(max_length=100)),
                ("color", models.CharField(max_length=100)),
                ("price", models.DecimalField(decimal_places=2, max_digits=5)),
            ],
        ),
    ]
