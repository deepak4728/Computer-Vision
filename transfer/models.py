from django.db import models

# Create your models here.

class ImageUpload(models.Model):
    content_image = models.ImageField(upload_to='content_images/')
    style_image = models.ImageField(upload_to='style_images/')
    output_image = models.ImageField(upload_to='output_images/', null=True, blank=True)
