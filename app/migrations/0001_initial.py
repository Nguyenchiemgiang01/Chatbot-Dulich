# Generated by Django 4.2 on 2023-10-14 18:41

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Conservation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_question', models.CharField(max_length=128, null=True)),
                ('bot_answer', models.CharField(max_length=128, null=True)),
            ],
        ),
    ]
