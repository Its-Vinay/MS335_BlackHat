# Generated by Django 3.0.8 on 2020-07-30 10:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('uniform', '0004_uniform_model_uni_output'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uniform_model',
            name='name',
            field=models.CharField(default='officer', max_length=20),
        ),
    ]
