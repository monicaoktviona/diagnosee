import requests
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from frontend.forms import Query

@csrf_exempt 
def frontend(request):
    return render(request, "index.html")

@csrf_exempt 
def search(request):
    if request.method=="POST":
        form = Query(request.POST)
        if form.is_valid():
            query = request.POST.get('query')
            hasil = get_serp(query)
            context={
                'query': query,
                'serp': hasil['serp'],
                'waktu': hasil['duration'],
                'length': hasil['length'],
            }
    return render(request, "result.html", context)
    
def get_serp(query):
        api_url = f"https://asia-southeast2-jarkom-rahma.cloudfunctions.net/diagnosee-search/search?query={query}"

        response = requests.get(api_url)

        if response.status_code == 200:
            api_data = response.json()
            return api_data
        else:
            error_message = "Failed to fetch data from the API"
            return JsonResponse({'error': error_message}, status=response.status_code)
        
