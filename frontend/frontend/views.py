import json
import requests
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.core import serializers

def frontend(request):
    return render(request, "base.html")

def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        serp = get_serp(query)
        return JsonResponse()
    return HttpResponse('')

def get_serp(query):
        api_url = f"https://asia-southeast2-jarkom-rahma.cloudfunctions.net/diagnosee-search/search?query={query}"

        response = requests.get(api_url)

        if response.status_code == 200:
            api_data = response.json()
            return api_data
        else:
            error_message = "Failed to fetch data from the API"
            return JsonResponse({'error': error_message}, status=response.status_code)
        
