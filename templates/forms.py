from django import forms
from .models import Patent, ClassificationResult

class PatentForm(forms.ModelForm):
    class Meta:
        model = Patent
        fields = ['application_number', 'application_date', 'publication_number', 'publication_date', 'cpc', 'ipc', 'claim', 'description']

class FeedbackForm(forms.ModelForm):
    class Meta:
        model = ClassificationResult
        fields = ['feedback']