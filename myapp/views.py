from django.shortcuts import render, get_object_or_404, redirect
from .models import Patent, ClassificationResult
from .forms import PatentForm, FeedbackForm
from .utils import classify_patent
from django.http import HttpResponse


def import_patent(request):
    if request.method == 'POST':
        form = PatentForm(request.POST)
        if form.is_valid():
            patent = form.save()
            classification_result = classify_patent(patent)
            return redirect('classification_result', pk=classification_result.pk)
    else:
        form = PatentForm()
    return render(request, 'myapp/import_patent.html', {'form': form})


def classification_result(request, pk):
    classification_result = get_object_or_404(ClassificationResult, pk=pk)
    if request.method == 'POST':
        form = FeedbackForm(request.POST, instance=classification_result)
        if form.is_valid():
            form.save()
            return redirect('classification_result', pk=pk)
    else:
        form = FeedbackForm(instance=classification_result)
    return render(request, 'myapp/classification_result.html',
                  {'classification_result': classification_result, 'form': form})


def home(request):
    return render(request, 'myapp/home.html')
