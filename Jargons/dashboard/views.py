from django.http import HttpResponseRedirect, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.urls.base import reverse
from django.contrib import messages
from django.shortcuts import render
import dashboard.Elastic as elastic


def index(request):
    message = None
    for msg in messages.get_messages(request):
        message = msg
        break
    return render(request, 'dashboard/home.html', {'message': message})


def upload(request):
    if request.method == "POST":
        try:
            url = "/api/v2/containers/"
            context = {}
            uploaded_file = request.FILES['csv-file']
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            print(name)
            context['url'] = fs.url(name)

            elastic.csv_writer(fs.path(name), str(name).split('.')[0].lower(), 'data', ';')

            messages.add_message(request, messages.INFO, 'File uploaded successfully')
            return HttpResponseRedirect(reverse('dashboard:index'))
        except Exception as e:
            print(e)
            messages.add_message(request, messages.INFO, 'Failed to upload file...!')
            return HttpResponseRedirect(reverse('dashboard:index'))

    message = None
    for msg in messages.get_messages(request):
        message = msg
        break
    return render(request, 'dashboard/home.html', {'message': message})


def indices(request):
    if request.method == 'GET':
        indices = list(elastic.get_indices())
        print('indices', indices)
        return JsonResponse({'indices': indices})

    return JsonResponse({})


def get_prediction(request):
    # if request.method == 'POST':
    #     index = request.POST.get('index')
    #
    pass
