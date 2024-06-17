from django.db import models


class Patent(models.Model):
    application_number = models.CharField(max_length=255)
    application_date = models.DateField()
    publication_number = models.CharField(max_length=255)
    publication_date = models.DateField()
    cpc = models.JSONField()
    ipc = models.JSONField()
    claim = models.TextField()
    description = models.TextField()

    def __str__(self):
        return self.application_number


class ClassificationResult(models.Model):
    patent = models.ForeignKey(Patent, on_delete=models.CASCADE)
    predicted_category = models.CharField(max_length=255)
    explanation = models.TextField()
    feedback = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f'{self.patent.application_number} - {self.predicted_category}'
