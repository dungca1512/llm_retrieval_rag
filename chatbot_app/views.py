from django.shortcuts import render
from django.http import JsonResponse
from src.query import answer_question
from src.config import Config


def chat_view(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        answer, context, metadatas = answer_question(query)
        return JsonResponse({'answer': answer, 'context': context, 'metadatas': metadatas})
    return render(request, 'chat.html')
